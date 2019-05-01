#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "map.hpp"
#include "checker.hpp"

#include <algorithm>
#include <climits>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <memory>


const int MAX_PROGRAM_SECS = 850;
int start_time = clock();

struct Scheduler {
    using PII = std::pair<int, int>;
    using PDI = std::pair<double, int>;
    using PID = std::pair<int, double>;

    const Map &mp;
    // max_per_ts: 每个时间片非预置车辆最多发车数
    // n_try: 二分发车最大次数
    int max_per_ts, n_try;
    // ahead: 预测未来路况
    // current: 维护当前时刻路况
    std::unique_ptr<Checker> ahead, current;
    // 定义时差为车辆真实用时-预计用时
    // mse: 本时刻出发车辆时差的均方误差
    // time_avg: 本时刻出发车辆真实用时的均值
    // time_avg_all: 已出发非预置车辆真实用时的均值
    // alpha: time_avg_all计算的衰减系数
    double mse = 0, time_avg = 0, time_avg_all = 0;
    static constexpr double alpha = 0.98;
    std::vector<double> car_time; // 车辆预估用时

    Scheduler(const Map &mp): mp(mp), car_time(mp.n_cars) {
        max_per_ts = n_try = 0;
        // 以道路数(含正反方向)作为max_per_ts的值
        for (const auto &road : mp.roads)
            max_per_ts += int(bool(road.bi) + 1);
        max_per_ts = std::max(256, max_per_ts);
        n_try = std::max(3, 32 - __builtin_clz(max_per_ts) - 6);
    }

    // 根据路况估计车辆行驶过某条道路时的行驶时间和cost
    // speed: 车辆速度
    // cross_i: 起点路口
    // road_i: 道路
    // shortest: cost是否用最短路
    std::pair<double, double> calc_dist(int speed, int cross_i, int road_i, int ts, bool shortest) {
        // 几个玄学参数
        // road_pow: 道路cost计算时随路上车辆数增长，略大于线性增长
        // road_cof, cross_cof: 对道路cost和路口cost的加权系数
        static constexpr double road_pow = 1.2, road_cof = 2, cross_cof = 0.01;
        const auto &road = mp.roads[road_i];
        bool is_forward = road.from == cross_i;

        double cross_bias = ahead->cross_bias[cross_i]; // 过路口的车辆平均时差
        int cnt1, cnt2;
        double road_speed1, road_speed2, road_bias1, road_bias2;
        double rspeed = std::min(speed, road.limit); // 车速在道路上的车速
        // 查询进入路口时的路况和预计出路口时刻的路况
        std::tie(cnt1, road_speed1, road_bias1) = ahead->get_info(road_i, is_forward, ts);
        std::tie(cnt2, road_speed2, road_bias2) = ahead->get_info(road_i, is_forward,
                                                                  ts + road.len / rspeed);
        if (road_speed1 < 1) road_speed1 = rspeed;
        if (road_speed2 < 1) road_speed2 = rspeed;
        double cnt = 0.5 * (cnt1 + cnt2);
        double road_speed = 0.5 * (road_speed1 + road_speed2);
        double road_bias = road_bias2;

        if (cnt * road_speed > road.len) // 车辆数较多，预估车速取道路平均车速
            rspeed = std::min(road_speed, rspeed);

        // 道路cost，玄学公式。让车辆数少，车道数多的道路cost更小
        double road_cost = pow(cnt, road_pow) / road.len / (road.num * road.num) * road_cof;
        // 路口cost，玄学公式。让分叉数多，车道数多的路口cost更大 (因为这样的路口更容易产生路口等待)
        double cross_cost = (mp.cross_nroads[cross_i] + sqrt( mp.cross_nlanes[cross_i])) * cross_cof;

        // 预估行驶时间
        double w = std::max(1.0, road.len / rspeed + road_bias + cross_bias);
        // 道路路口的cost与行驶时间进行加权的最终cost。玄学公式
        double c = (road_cost + cross_cost) / rspeed * time_avg_all + w;
        if (shortest) c = w;
        return std::make_pair(w, c);
    }

    // 预估一辆车既定路线的用时和cost
    void estimate(int car_i, Schedule &sch) {
        const auto &car = mp.cars[car_i];
        int u = car.from;
        double dist = 0., e = 0., w, c;
        for (auto &p : sch.Path) {
            int road_i = p.first;
            const auto &road = mp.roads[road_i];
            int v = u == road.from ? road.to : road.from;
            int t = std::max(0, int(dist + 0.5) + sch.start);
            std::tie(w, c) = calc_dist(car.speed, u, road_i, t, false);
            dist += w, e += c, u = v;
            p.second = w;
        }
        assert(u == car.to);
        car_time[car_i] = dist;
    }

    // 按照路径cost计算最短路
    std::tuple<std::vector<double>, std::vector<double>, std::vector<PID>>
    dijkstra(int from, int speed, int ts, int to, bool shortest) {
        static constexpr double INF = 1LL << 60;
        std::vector<PID> Pre(mp.n_crosses);
        std::priority_queue<PDI, std::vector<PDI>, std::greater<PDI>> PQ;
        std::vector<double> Dist(mp.n_crosses, INF), E(mp.n_crosses, INF);
        PQ.emplace(0, from), Dist[from] = 0, E[from] = 0;
        std::vector<char> Closed(mp.n_cars, 0);
        while (!PQ.empty()) {
            int u;
            double eu;
            std::tie(eu, u) = PQ.top(), PQ.pop();
            if (to != -1 && u == to) break;
            if (Closed[u]) continue;
            Closed[u] = 1;
            for (const auto &p : mp.G[u]) {
                int v = p.first, road_i = p.second;
                if (Closed[v]) continue;
                if (mp.roads[road_i].limit == 0) continue;
                double w, c;
                int t = std::max(0, int(Dist[u] + 0.5) + ts);
                std::tie(w, c) = calc_dist(speed, u, road_i, t, shortest);
                if (E[u] + c < E[v]) {
                    E[v] = E[u] + c, Dist[v] = Dist[u] + w, Pre[v] = {road_i, w};
                    PQ.emplace(E[v], v);
                }
            }
        }
        return std::make_tuple(Dist, E, Pre);
    }

    // 生成路径
    void get_path(int car_i, std::vector<PID> &Path, int ts, bool shortest) {
        const auto &car = mp.cars[car_i];
        std::vector<PID> Pre;
        std::vector<double> Dist, E;
        std::tie(Dist, E, Pre) = dijkstra(car.from, car.speed, ts, car.to, shortest);
        int tmp = car.to;
        Path.clear();
        while (tmp != car.from) {
            const auto &road = mp.roads[Pre[tmp].first];
            Path.emplace_back(Pre[tmp]);
            tmp = road.from == tmp ? road.to : road.from;
        }
        reverse(Path.begin(), Path.end());
        car_time[car_i] = Dist[car.to];
    }

    // 批量预估车辆用时
    bool estimate_time_multi(const std::vector<int> &I, int ts, bool shortest) {
        static const int MAX_TURN = 3000;
        if (mp.n_crosses > MAX_TURN) return false;
        std::map<int, int> speed_cnt;
        std::vector<int> Speed;
        for (auto i : I) speed_cnt[mp.cars[i].speed]++;
        while (speed_cnt.size() * mp.n_crosses > MAX_TURN) speed_cnt.erase(speed_cnt.begin());
        for (auto p : speed_cnt) Speed.push_back(p.first);
        // 按照车速和出发路口将车辆划分
        std::vector<std::vector<int>> SCI(Speed.size() * mp.n_crosses);
        for (auto i : I) {
            int si = std::lower_bound(Speed.begin(), Speed.end(), mp.cars[i].speed) - Speed.begin();
            int idx = si * mp.n_crosses + mp.cars[i].from;
            SCI[idx].push_back(i);
        }
        for (size_t i = 0; i < Speed.size(); i++) {
            int speed = Speed[i];
            for (int from = 0; from < mp.n_crosses; from++) {
                std::vector<PID> Pre;
                std::vector<double> Dist, E;
                std::tie(Dist, E, Pre) = dijkstra(from, speed, ts, -1, shortest);
                for (auto car_i : SCI[i * mp.n_crosses + from]) {
                    int to = mp.cars[car_i].to;
                    car_time[car_i] = Dist[to];
                }
            }
        }
        return true;
    }

    // 修改部分预置车辆的实际出发时间或路径
    // 返回预置车辆的ts, tpri和te
    std::tuple<int, int, int> change_preset(std::vector<Schedule> &Sch, int max_ts = 100000,
                                            double portion = 0.1) {
        enum { CHANGE_TIME, CHANGE_PATH };
        assert(portion >= 0.0 && portion <= 1.0);
        // 找出所有的预置车辆
        std::vector<int> I;
        for (int i = 0; i < mp.n_cars; i++) if (mp.cars[i].preset) I.push_back(i);
        std::vector<int> arrive_true(mp.n_cars), arrive_change_time(mp.n_cars),
            arrive_change_path(mp.n_cars), arrive_estimate(mp.n_cars), Label(mp.n_cars);

        ahead = std::unique_ptr<Checker>(new Checker(mp));
        ahead->run(Sch, I, max_ts);
        for (auto i : I) arrive_true[i] = ahead->car_time[i] + Sch[i].start;
        for (auto i : I) arrive_change_time[i] = ahead->car_time[i] + mp.cars[i].start;

        ahead = std::unique_ptr<Checker>(new Checker(mp));
        std::vector<PID> dummy_path;
        for (auto i : I) {
            get_path(i, dummy_path, Sch[i].start, mp.cars[i].prior);
            arrive_change_path[i] = car_time[i] + Sch[i].start;
        }

        for (auto i : I) {
            Label[i] = arrive_change_path[i] < arrive_change_time[i] ? CHANGE_PATH : CHANGE_TIME;
            arrive_estimate[i] = std::min(arrive_change_path[i], arrive_change_time[i]);
        }
        // 按照节省的时间排序
        auto cmp = [&](int i, int j) {
            int deltai = arrive_true[i] - arrive_estimate[i];
            int deltaj = arrive_true[j] - arrive_estimate[j];
            int savei = mp.cars[i].prior ? (mp.a_ + 1) * deltai : deltai;
            int savej = mp.cars[j].prior ? (mp.a_ + 1) * deltaj : deltaj;
            return savei > savej;
        };
        std::stable_sort(I.begin(), I.end(), cmp);
        int l = 0, r = int(portion * I.size()) + 1;
        auto backup1 = Sch;
        while (l + 1 < r) {
            int m = (l + r) / 2;
            auto Tmp = backup1;
            for (int i = 0; i < m; i++) {
                int car_i = I[i];
                if (Label[car_i] == CHANGE_PATH) {
                    get_path(car_i, Tmp[car_i].Path, Tmp[car_i].start, true);
                } else Tmp[car_i].start = mp.cars[car_i].start;
                Tmp[car_i].change = 1;
            }
            int s = std::unique_ptr<Checker>(new Checker(mp))->run(Tmp, I, max_ts);
            if (s == FINISH) backup1 = Tmp, l = m;
            else r = m;
        }
        printf("预置车辆修改路径%d条\n", l);
        auto checker = std::unique_ptr<Checker>(new Checker(mp));
        checker->run(backup1, I, max_ts);
        int ts1, tpri1, te1;
        std::tie(ts1, tpri1, te1) = std::make_tuple(checker->ts, checker->tpri, checker->te);

        // 按照预计到达时间排序
        auto cmp2 = [&](int i, int j) {
            int arrivei = mp.cars[i].prior ? (mp.a_ + 1) * (arrive_true[i] - mp.early_prior) :
                          arrive_true[i];
            int arrivej = mp.cars[j].prior ? (mp.a_ + 1) * (arrive_true[j] - mp.early_prior) :
                          arrive_true[j];
            return arrivei > arrivej;
        };
        auto backup2 = Sch;
        std::stable_sort(I.begin(), I.end(), cmp2);
        while (l + 1 < r) {
            int m = (l + r) / 2;
            auto Tmp = backup2;
            for (int i = 0; i < m; i++) {
                int car_i = I[i];
                if (Label[car_i] == CHANGE_PATH) {
                    get_path(car_i, Tmp[car_i].Path, Tmp[car_i].start, true);
                } else Tmp[car_i].start = mp.cars[car_i].start;
                Tmp[car_i].change = 1;
            }
            int s = std::unique_ptr<Checker>(new Checker(mp))->run(Tmp, I, max_ts);
            if (s == FINISH) backup2 = Tmp, l = m;
            else r = m;
        }
        printf("预置车辆修改路径%d条\n", l);
        checker = std::unique_ptr<Checker>(new Checker(mp));
        checker->run(backup2, I, max_ts);
        int ts2, tpri2, te2;
        std::tie(ts2, tpri2, te2) = std::make_tuple(checker->ts, checker->tpri, checker->te);

        // 比较两种方案，采用使预置车辆te最小的方案
        if (te1 < te2) return Sch = backup1, std::make_tuple(ts1, tpri1, te1);
        else return Sch = backup2, std::make_tuple(ts2, tpri2, te2);
    }


    // margin: 优先车辆Tpri允许比预计值延后的范围，-1不限制
    // 如果成功生成，返回ts, tpri和te
    std::tuple<int, int, int> gen_schedule(std::vector<Schedule> &Sch, int margin = -1,
                                           int max_ts = 100000, int max_te = INT_MAX, bool print = true) {
        time_avg = time_avg_all = mse = 0.0; // 重置统计量

        std::vector<int> P; // 当前时刻还没有出发的预置车辆
        // In[ts], PreIn[ts]: 计划出发时间为ts的普通/预置车辆集合
        std::vector<std::vector<int>> In(max_ts), PreIn(max_ts);
        for (int i = 0; i < mp.n_cars; i++) {
            if (mp.cars[i].preset) PreIn[Sch[i].start].push_back(i);
            else In[mp.cars[i].start].push_back(i);
        }
        // P按时刻排序,出发时刻早的排在末尾
        for (int t = max_ts - 1; t >= 0; t--) for (auto i : PreIn[t]) P.push_back(i);

        // 生成只跑预置车辆的路况
        ahead = std::unique_ptr<Checker>(new Checker(mp));
        int s = ahead->run(Sch, P, max_ts);
        if (s == TIMEUP) throw std::logic_error("预设车辆系统调度超时");
        if (s == DEADLOCK) throw std::logic_error("预设车辆死锁");
        if (print) {
            printf("预置车辆T: %d\n", ahead->ts);
            printf("预置车辆Te: %d\n", ahead->te);
        }

        // 预计ts和tpri的最小值
        int init_ts = ahead->ts, init_tpri = ahead->tpri;
        for (int i = 0; i < mp.n_cars; i++) {
            if (mp.cars[i].preset) car_time[i] = ahead->car_time[i];
            else {
                get_path(i, Sch[i].Path, mp.cars[i].start, false);
                init_ts = std::max(init_ts, int(car_time[i] + mp.cars[i].start + 0.5));
                if (mp.cars[i].prior)
                    init_tpri = std::max(init_tpri, int(car_time[i] + mp.cars[i].start + 0.5 - mp.early_prior));
            }
        }
        if (print) {
            printf("预计最小T: %d\n", init_ts);
            printf("预计最小Tpri: %d\n", init_tpri);
        }

        // 按时间片发车
        current = std::unique_ptr<Checker>(new Checker(mp)); // 重置当前路况
        int n_sent = 0, n_sent_p = 0; // 已发车数

        for (int ts = 0; n_sent < mp.n_cars && ts < max_ts; ts++) {
            int cur = clock();
            int program_secs = double(cur - start_time) / CLOCKS_PER_SEC + 0.5;
            if (program_secs > MAX_PROGRAM_SECS) throw std::logic_error("程序运行超时");
            if (print) printf("程序已用时: %d secs\n", program_secs);

            // 确定能够发车的数量
            int sz = int(In[ts].size());
            // int l = 0, r = std::max(0, std::min(max_per_ts, sz) - int(PreIn[ts].size())) + 1;
            int l = 0, r = std::max(0, std::min(max_per_ts, sz)) + 1;
            if (print) printf("最多发车数: %d\n", r - 1);

            // 确定发车顺序
            estimate_time_multi(In[ts], ts, false); // 预估所有待发车辆的用时
            // 按照 是否是优先车辆，是否可能对te造成影响，车速与预计用时进行排序
            int prior_t = std::max(init_tpri, ahead->tpri) + mp.early_prior - 10;
            int all_t = std::max(init_ts, ahead->ts) - 10;
            stable_sort(In[ts].begin(), In[ts].end(), [&](int i, int j) {
                const Car &ci = mp.cars[i], &cj = mp.cars[j];
                if (ci.prior != cj.prior) return ci.prior < cj.prior;
                bool b1 = car_time[i] + ts > (ci.prior ? prior_t : all_t);
                bool b2 = car_time[j] + ts > (cj.prior ? prior_t : all_t);
                if (b1 != b2) return b1 < b2;
                return std::make_tuple(-ci.speed, car_time[i]) <
                       std::make_tuple(-cj.speed, car_time[j]);
            });

            // 规划候选待车辆的路径
            for (int i = 0; i < r - 1; i++) {
                int car_i = In[ts][sz - i - 1];
                Sch[car_i].start = ts;
                get_path(car_i, Sch[car_i].Path, ts, false);
            }

            bool flag = false; // 是否可以成功发车
            std::unique_ptr<Checker> backup;
            // 判断优先级最高的m辆车是否可以在本时刻发车
            auto check = [&](int m) {
                if (!m) return true;
                std::unique_ptr<Checker> tmp(new Checker(*current));
                for (int j = 0; j < m; j++) P.push_back(In[ts][sz - 1 - j]);
                int s = tmp->run(Sch, P, max_ts);
                for (int j = 0; j < m; j++) P.pop_back();
                bool ret = s == FINISH;
                if (margin >= 0 && tmp->tpri > init_tpri + margin) return false;
                if (ret) {
                    flag = true;
                    backup = std::unique_ptr<Checker>(new Checker(*tmp));
                }
                return ret;
            };
            if (check(r - 1)) {
                l = r - 1;
            } else {
                int n_bs = 0;
                while (l + 1 < r) {
                    if (n_bs++ == n_try) break;
                    int m = (l + r) >> 1;
                    if (!m) break;
                    if (check(m)) l = m;
                    else r = m;
                }
            }
            // 防止候选车辆内部造成死锁
            if (l == 0 && r > 1 && current->n == current->n_arrive) if (check(1)) l = 1;

            // 更新未来路况
            if (flag) ahead = std::unique_ptr<Checker>(new Checker(*backup));
            if (print) {
                printf("%d: %d / %d, %d / %d\n", ts, l,
                       sz, int(PreIn[ts].size()), int(P.size()));
                ahead->print();
            }

            // 当前时刻发车的车辆
            std::vector<int> Add;
            // 本时刻发车的非预置车辆加入已发车列表
            for (int i = 0; i < l; i++) Add.push_back(In[ts].back()), In[ts].pop_back();
            // 未发车辆移动到下一时刻的待发车辆列表中
            if (ts + 1 < max_ts) {
                if (In[ts].size() > In[ts + 1].size()) swap(In[ts], In[ts + 1]);
                for (auto car_i : In[ts]) In[ts + 1].push_back(car_i);
                In[ts].clear();
            }
            // 本时刻预设车辆加入已发车列表中
            while (!P.empty() && Sch[P.back()].start == ts)
                Add.push_back(P.back()), P.pop_back();

            // 更新各统计量
            if (!Add.empty()) for (int i = 0, nn = 0, sz = int(Add.size()); i < sz; i++) {
                    int car_i = Add[i];
                    if (current->car_time[car_i]) continue;
                    if (mp.cars[car_i].preset) continue;
                    if (!nn) mse =  time_avg = 0;
                    nn++;
                    double x = ahead->car_time[car_i];
                    double d = ahead->car_time[car_i] - car_time[car_i];
                    time_avg += (x - time_avg) / nn;
                    time_avg_all = time_avg_all < 1 ? x : time_avg_all * alpha + (1 - alpha) * x;
                    mse += (d * d - mse) / nn;
                }

            current->run(Sch, Add, ts + 1); // 车辆上路，更新当前路况
            n_sent += Add.size();
            for (auto i : Add) if (mp.cars[i].prior) n_sent_p++;

            if (margin >= 0 && ts > init_tpri + margin + mp.early_prior && current->n_p < mp.n_prior)
                throw std::logic_error("优先车辆在时限内未全部发车");
            if (ahead->te > max_te) throw std::logic_error("超过te上限");

            if (print) {
                printf("用时偏差RMSE / 用时均值: %.3f / %.3f = %.3f\n", sqrt(mse), time_avg,
                       sqrt(mse) / std::max(1.0, time_avg));
                printf("car_time: ");
                for (int i = 0; i < std::min(5, l); i++) printf("%.3f ", car_time[Add[i]]);
                printf("\n路上车辆数量: %d (%d)\n", current->n - current->n_arrive,
                       current->n_p - current->n_arrive_p);
                printf("运行用时: %.1f\n\n", double(clock() - cur) * 1000 / CLOCKS_PER_SEC), cur = clock();
            }
        }
        if (n_sent < mp.n_cars) throw std::logic_error("规定时间内未全部发车");
        ahead->print();
        return std::make_tuple(ahead->ts, ahead->tpri, ahead->te);
    }
};


#endif /* ifndef SCHEDULER_H */
