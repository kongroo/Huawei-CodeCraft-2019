#ifndef CHECKER_H
#define CHECKER_H

#include "map.hpp"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <functional>
#include <limits>
#include <list>
#include <tuple>
#include <unordered_set>

enum { WAIT, END }; // 车辆状态
enum { DEADLOCK, FINISH, OK, TIMEUP }; // 系统状态
enum { SUCCESS, FAIL, FULL, LIMIT, GOAL }; // 尝试行驶的结果


struct CarInfo {
    // 行驶车辆的状态
    // rd: 车辆目前所在道路在其规划路径上的位置(0下标)
    // pos: 对路上车辆表示其位置，合法值为从1到道路长度
    // lane: 车辆当前所在的车道编号
    // state: 车辆状态，等待或终止
    // dir: 车辆当前方向
    // ts_on: 车辆驶入当前道路的时刻
    // ts_at: 车辆准备过路口的时刻
    int prior, speed, rd, pos, lane, ts_on, ts_at;
    char state, dir;
    const Car &car;
    const Schedule &sch;
    const Map &mp;
    CarInfo(const Car &car, const Schedule &sch, const Map &mp) :
        prior(car.prior), speed(car.speed), car(car), sch(sch), mp(mp) {}
    // 找到车辆的行进方向
    int get_dir() {
        const auto &Path = sch.Path;
        if (rd + 1 == int(Path.size())) return FIN;
        int now_rd = Path[rd].first, nxt_rd = Path[rd + 1].first;
        int a = mp.roads[now_rd].from, b = mp.roads[now_rd].to,
            c = mp.roads[nxt_rd].from, d = mp.roads[nxt_rd].to;
        int cross_i = (a == c || a == d) ? a : b;
        int i = mp.RI[cross_i][now_rd], j = mp.RI[cross_i][nxt_rd];
        int x = (j - i + 4) & 3;
        return x == 1 ? LEFT : x == 2 ? STRAIGHT : RIGHT;
    }
};

struct Checker {
    const Map &mp;
    const double alpha = 0.95;
    int n, n_p, n_pre, n_pre_p, n_arrive, n_arrive_p, n_sent, n_wait;
    int ts, tpri, tsum, tsumpri, te, tesum, prev_ts = 0;

    // 对已达终点的车辆，表示其实际调度时间
    std::vector<int> car_time;
    // 道路正反方向上每个时刻的车辆数，平均车速和用时偏差
    std::vector<std::vector<std::tuple<int, int, double, double>>> road_info[2];
    // 维护当前时刻道路上的车辆数
    std::vector<int> road_cnt_cur[2];
    // 维护当前时刻道路上的平均车速
    std::vector<double> road_speed_cur[2];
    // 道路上车辆的真实用时和预测用时的平均差值
    std::vector<double> road_bias[2];
    // 路口平均延时，偷懒没有分时间片记录
    std::vector<int> cross_pass_cnt;
    std::vector<double> cross_bias;
    // 每个时刻预计出发的车辆
    std::vector<std::vector<int>> start_ts;
    // 当前时刻每条道路上车库中待出发的车辆
    std::vector<std::list<CarInfo *>> ready[2][2];
    // 正/反向道路车道
    std::vector<std::vector<std::list<CarInfo *>>> car_roads[2];

    Checker(const Map &mp) : mp(mp) {
        for (int is_forward : {0, 1}) {
            car_roads[is_forward].resize(mp.n_roads);
            for (int i = 0; i < mp.n_roads; i++) {
                car_roads[is_forward][i].resize(mp.roads[i].num);
            }
        }
        n = n_p = n_pre = n_pre_p =  n_arrive = n_arrive_p = n_sent = ts = tsum = tpri = tsumpri = 0;
        car_time.assign(mp.n_cars, 0);
        for (int is_forward : {0, 1}) {
            road_info[is_forward].resize(mp.n_roads);
            road_cnt_cur[is_forward].assign(mp.n_roads, 0);
            road_speed_cur[is_forward].assign(mp.n_roads, 0.0);
            road_bias[is_forward].assign(mp.n_roads, 0.0);
            ready[is_forward][0].resize(mp.n_roads);
            ready[is_forward][1].resize(mp.n_roads);
            for (int i = 0; i < mp.n_roads; i++) {
                road_info[is_forward][i].assign(1, {-1, 0, 0.0, 0.0});
                ready[is_forward][0][i].clear(), ready[is_forward][1][i].clear();
            }
        }
        cross_pass_cnt.assign(mp.n_crosses, 0);
        cross_bias.assign(mp.n_crosses, 0.0);
    }

    Checker(const Checker &o): mp(o.mp), n(o.n), n_p(o.n_p), n_pre(o.n_pre), n_pre_p(o.n_pre_p),
        n_arrive(o.n_arrive), n_arrive_p(o.n_arrive_p), n_sent(o.n_sent), n_wait(o.n_wait),
        ts(o.ts), tpri(o.tpri), tsum(o.tsum), tsumpri(o.tsumpri), te(o.te), tesum(o.tesum),
        car_time(o.car_time), start_ts(o.start_ts) {
        auto copy_list = [](std::list<CarInfo *> &to, const std::list<CarInfo *> &from) {
            to.resize(from.size());
            auto it1 = to.begin();
            auto it2 = from.begin();
            for (; it1 != to.end(); ++it1, ++it2)
                *it1 = new CarInfo(**it2);
        };
        for (int is_forward : {0, 1}) {
            road_info[is_forward] = o.road_info[is_forward];
            road_cnt_cur[is_forward] = o.road_cnt_cur[is_forward];
            road_speed_cur[is_forward] = o.road_speed_cur[is_forward];
            road_bias[is_forward] = o.road_bias[is_forward];
            car_roads[is_forward] = o.car_roads[is_forward];
            for (size_t i = 0; i < car_roads[is_forward].size(); i++)
                for (size_t j = 0; j < car_roads[is_forward][i].size(); j++)
                    copy_list(car_roads[is_forward][i][j], o.car_roads[is_forward][i][j]);
        }
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++) {
                ready[i][j].resize(o.ready[i][j].size());
                for (size_t k = 0; k < ready[i][j].size(); k++)
                    copy_list(ready[i][j][k], o.ready[i][j][k]);
            }
        cross_pass_cnt = o.cross_pass_cnt;
        cross_bias = o.cross_bias;
    }

    ~Checker() {
        auto delete_list = [](std::list<CarInfo *> &L) {
            while (!L.empty()) delete L.back(), L.pop_back();
        };
        for (int i = 0; i < 2; i++) for (int j = 0; j < 2; j++)
                for (auto &L : ready[i][j]) delete_list(L);
        for (int is_forward : {0, 1}) for (auto &car_road : car_roads[is_forward])
                for (auto &lane : car_road) delete_list(lane);
    }

    void print() {
        printf("车辆数: %d / %d (%d / %d)\n", n, mp.n_cars, n_p, mp.n_prior);
        printf("T: %d, Tsum: %d\n", ts, tsum);
        printf("Tpri: %d, Tsumpri: %d\n", tpri, tsumpri);
        printf("Te: %d, Tesum: %d\n", te, tesum);
        fflush(stdout);
    }

    // 车道内调度
    void do_lane(const std::list<CarInfo *> &lane, const Road &road, bool head, bool init) {
        if (lane.empty() || (lane.size() < 2 && !head)) return;
        auto it = lane.begin();
        int pre_pos = road.len, pre_state = WAIT;
        if (!head) pre_pos = (*it)->pos - 1, pre_state = (*it)->state, ++it;
        for (bool ok; it != lane.end(); ++it) {
            auto p = *it;
            if (p->state == END) break;
            int speed = std::min(p->speed, road.limit);
            ok = true;
            if (pre_pos - p->pos >= speed) p->pos += speed;
            else if (pre_state != WAIT) p->pos = pre_pos;
            else ok = false;
            if (!ok && !init) break;
            if (ok) p->state = END, n_wait--;
            pre_pos = p->pos - 1, pre_state = p->state;
        }
    }

    // 找到道路road_i上即将通过路口cross_i的第一优先级的车辆
    CarInfo *get_first(int cross_i, int road_i) {
        if (road_i == -1) return nullptr;
        const auto &road = mp.roads[road_i];
        bool is_forward = cross_i == road.to;
        if (!is_forward && !road.bi) return nullptr;
        if (road_cnt_cur[is_forward][road_i] == 0) return nullptr;
        const auto &car_road = car_roads[is_forward][road_i];
        int pos = -1, prior = -1;
        CarInfo *c = nullptr;
        for (const auto &lane : car_road) {
            if (lane.empty()) continue;
            const auto cp = lane.front();
            if (cp->state == END) continue;
            if (std::tie(cp->prior, cp->pos) > std::tie(prior, pos))
                prior = cp->prior, pos = cp->pos, c = cp;
        }
        return c;
    }

    // 尝试将车辆cp从道路ri1行驶到道路ri2
    int drive(CarInfo *cp, int ri1, int ri2, int cross_i) {
        int s2;
        bool is_forward1 = (ri1 != -1 && mp.roads[ri1].to == cross_i);
        bool is_forward2 = (ri2 != -1 && mp.roads[ri2].from == cross_i);

        if (ri2 == -1) { // 行驶到终点
            assert(ri1 != -1);
            // 驶出ri1
            auto &lane = car_roads[is_forward1][ri1][cp->lane];
            lane.pop_front(), do_lane(lane, mp.roads[ri1], true, false);
            //更新车辆相关统计量
            n_arrive++, n_wait--, n_arrive_p += bool(cp->prior);
            tsum += ts - cp->car.start, car_time[cp->car.i] = ts - cp->sch.start;
            if (cp->prior) tpri = ts - mp.early_prior, tsumpri += ts - cp->car.start;
            // 更新道路相关统计量
            int &rcnt = road_cnt_cur[is_forward1][ri1];
            rcnt--;
            double &rspeed = road_speed_cur[is_forward1][ri1];
            if (rcnt == 0) rspeed = 0.0;
            else rspeed -= (std::min(cp->speed, mp.roads[ri1].limit) - rspeed) / rcnt;
            if (!cp->car.preset) {
                auto &bias = road_bias[is_forward1][ri1];
                double d = ts - cp->ts_on - cp->sch.Path.back().second;
                bias = bias * alpha + d * (1 - alpha);
            }
            // 更新路口相关统计量
            int &ccnt = cross_pass_cnt[cross_i];
            ccnt++;
            double &cbias = cross_bias[cross_i];
            cbias += (ts - cp->ts_at - cbias) / ccnt;
            return delete cp, GOAL;
        }

        // 计算s2
        if (ri1 == -1) s2 = std::min(cp->speed, mp.roads[ri2].limit);
        else {
            int v2 = std::min(mp.roads[ri2].limit, cp->speed);
            int s1 = mp.roads[ri1].len - cp->pos;
            s2 = v2 - s1;
        }

        if (s2 <= 0) { // 因限速规则无法行驶到ri2
            assert(ri1 != -1);
            cp->state = END, cp->pos = mp.roads[ri1].len, n_wait--;
            auto &lane = car_roads[is_forward1][ri1][cp->lane];
            return do_lane(lane, mp.roads[ri1], false, false), LIMIT;
        }

        // 枚举车道尝试上路
        auto &lanes = car_roads[is_forward2][ri2];
        for (int i = 0; i < (int)lanes.size(); i++) {
            int front_pos = lanes[i].empty() ? std::numeric_limits<int>::max() : lanes[i].back()->pos;
            if (front_pos > s2) {
                cp->pos = std::min(s2, mp.roads[ri2].len);
            } else if (lanes[i].back()->state == WAIT) {
                return FAIL;
            } else if (front_pos > 1) {
                cp->pos = front_pos - 1;
            } else continue;
            // 行驶到ri2上
            if (ri1 != -1)  {
                // 驶出ri1
                auto &lane = car_roads[is_forward1][ri1][cp->lane];
                lane.pop_front(), do_lane(lane, mp.roads[ri1], true, false);
                // 更新车辆相关统计量
                cp->rd++, n_wait--;
                // 更新道路相关统计量
                int &rcnt1 = road_cnt_cur[is_forward1][ri1];
                rcnt1--;
                double &rspeed1 = road_speed_cur[is_forward1][ri1];
                if (rcnt1 == 0) rspeed1 = 0.0;
                else rspeed1 -= (std::min(cp->speed, mp.roads[ri1].limit) - rspeed1) / rcnt1;
                if (!cp->car.preset) {
                    double &bias = road_bias[is_forward1][ri1];
                    double d = ts - cp->ts_on - cp->sch.Path[cp->rd - 1].second;
                    bias = bias * alpha + d * (1 - alpha);
                }
            } else n_sent++, cp->rd = 0;
            // 更新车辆状态
            cp->state = END, cp->lane = i, cp->dir = cp->get_dir(), cp->ts_on = ts;
            // 驶入ri2
            lanes[i].push_back(cp);
            // 更新道路相关统计量
            int &rcnt2 = road_cnt_cur[is_forward2][ri2];
            double &speed2 = road_speed_cur[is_forward2][ri2];
            rcnt2++, speed2 += (std::min(cp->speed, mp.roads[ri2].limit) - speed2) / rcnt2;
            // 更新路口相关统计量
            int &ccnt = cross_pass_cnt[cross_i];
            ccnt++;
            double &cbias = cross_bias[cross_i];
            cbias += (ts - cp->ts_at - cbias) / ccnt;
            cp->ts_at = -1;
            return SUCCESS;
        }
        // 所有车道均被占满且车道尾部的车辆都是终止态
        cp->state = END;
        if (ri1 != -1) {
            cp->pos = mp.roads[ri1].len, n_wait--;
            auto &lane = car_roads[is_forward1][ri1][cp->lane];
            do_lane(lane, mp.roads[ri1], false, false);
        }
        return FULL;
    }

    int run_step(const std::vector<Schedule> &Sch) {
        // 添加当前时刻预计出发车辆到车库中
        sort(start_ts[ts].begin(), start_ts[ts].end());
        for (auto car_i : start_ts[ts]) {
            int road_i = Sch[car_i].Path.front().first;
            const auto &car = mp.cars[car_i];
            bool is_forward = car.from == mp.roads[road_i].from;
            auto cp = new CarInfo(car, Sch[car.i], mp);
            cp->ts_at = ts;
            ready[is_forward][car.prior][road_i].push_back(cp);
        }
        start_ts[ts].clear();
        std::vector<char> C(mp.n_crosses, 1); // 路口是否需要遍历
        auto drive_ready = [&](int road_i, bool is_forward, bool is_prior) {
            auto &L = ready[is_forward][is_prior][road_i];
            int cross_i = is_forward ? mp.roads[road_i].from : mp.roads[road_i].to;
            for (auto it = L.begin(); it != L.end();) {
                int s = drive(*it, -1, road_i, cross_i);
                if (s == FULL) break;
                if (s == SUCCESS) {
                    L.erase(it++);
                    C[cross_i] = 1;
                } else ++it;
            }
        };
        // 调度不过路口的车
        n_wait = 0;
        for (int road_i = 0; road_i < mp.n_roads; road_i++) {
            auto &road = mp.roads[road_i];
            for (int i = 0; i < mp.roads[road_i].num; i++) {
                for (auto cp : car_roads[1][road_i][i])
                    cp->state = WAIT, n_wait++;
                do_lane(car_roads[1][road_i][i], road, true, true);
                if (mp.roads[road_i].bi) {
                    for (auto cp : car_roads[0][road_i][i])
                        cp->state = WAIT, n_wait++;
                    do_lane(car_roads[0][road_i][i], road, true, true);
                }
            }
            drive_ready(road_i, true, true);
            if (mp.roads[road_i].bi) drive_ready(road_i, false, true);
        }
        // 更新路上车辆数和车速
        for (int i = 0; i < mp.n_roads; i++) {
            if (abs(road_cnt_cur[1][i] - std::get<1>(road_info[1][i].back())) > 0)
                road_info[1][i].emplace_back(ts, road_cnt_cur[1][i], road_speed_cur[1][i], road_bias[1][i]);
            if (mp.roads[i].bi) {
                if (abs(road_cnt_cur[0][i] - std::get<1>(road_info[0][i].back())) > 0)
                    road_info[0][i].emplace_back(ts, road_cnt_cur[0][i], road_speed_cur[0][i], road_bias[0][i]);
            }
        }
        // 调度路口中的车
        std::vector<std::vector<CarInfo *>> First(mp.n_crosses, std::vector<CarInfo *>(4, nullptr));
        // 计算路口各道路当前第一优先级车辆
        for (int i = 0; i < mp.n_crosses; i++) {
            for (int j = 0; j < 4; j++) {
                auto cp = get_first(i, mp.crosses[i].roads[j]);
                if (cp != nullptr && cp->ts_at == -1) cp->ts_at = ts;
                First[i][j] = cp;
            }
        }
        for (bool ok = false; !ok;) {
            ok = true;
            for (int cross_i = 0; cross_i < mp.n_crosses; cross_i++) {
                if (!C[cross_i]) continue;
                C[cross_i] = 0;
                for (int road_i : mp.InG[cross_i]) {
                    const auto &road = mp.roads[road_i];
                    bool is_forward = cross_i == road.to;
                    for (;;) {
                        auto &FC = First[cross_i];
                        const auto &RIC = mp.RI[cross_i];
                        int ri = RIC[road_i];
                        auto cp = FC[RIC[road_i]];
                        if (cp == nullptr) break;
                        const auto &TurnCR = mp.Turn[ri][cross_i];
                        int dir = cp->dir, road_i2 = TurnCR[dir];
                        // 行驶方向优先级冲突判断
                        int prior = cp->prior;
                        if (dir >= STRAIGHT) {
                            int rl = TurnCR[LEFT], rr = TurnCR[RIGHT];
                            auto cl = rl == -1 ? nullptr : FC[RIC[rl]], cr = rr == -1 ? nullptr : FC[RIC[rr]];
                            if (cl && cl->prior > prior && cl->dir == LEFT) break;
                            if (cr && cr->prior > prior && cr->dir == RIGHT) break;
                        } else if (dir == LEFT) {
                            int rr = TurnCR[RIGHT], ro = TurnCR[STRAIGHT];
                            auto co = ro == -1 ? nullptr : FC[RIC[ro]], cr = rr == -1 ? nullptr : FC[RIC[rr]];
                            if (cr && cr->prior >= prior && cr->dir >= STRAIGHT) break;
                            if (co && co->prior > prior && co->dir == RIGHT) break;
                        } else if (dir == RIGHT) {
                            int rl = TurnCR[LEFT], ro = TurnCR[STRAIGHT];
                            auto cl = rl == -1 ? nullptr : FC[RIC[rl]], co = ro == -1 ? nullptr : FC[RIC[ro]];
                            if (cl && cl->prior >= prior && cl->dir >= STRAIGHT) break;
                            if (co && co->prior >= prior && co->dir == LEFT) break;
                        }
                        // 没有冲突，尝试行车
                        int t = drive(cp, road_i, road_i2, cross_i);
                        if (t == FAIL) break;
                        C[road.from] = C[road.to] = 1;
                        ok = false;
                        auto ncp = get_first(cross_i, road_i);
                        if (ncp != nullptr && ncp->ts_at == -1) ncp->ts_at = ts;
                        FC[ri] = ncp;
                        // 优先车辆上路
                        drive_ready(road_i, is_forward, true);
                    }
                }
            }
        }
        if (n_wait) return DEADLOCK;
        // 车库中的车上路
        for (int i = 0; i < mp.n_roads; i++) {
            drive_ready(i, true, true), drive_ready(i, true, false);
            if (mp.roads[i].bi)
                drive_ready(i, false, true), drive_ready(i, false, false);
        }
        if (n_arrive == n) return FINISH;
        return OK;
    }

    int run(const std::vector<Schedule> &Sch, const std::vector<int> &I, int max_ts) {
        prev_ts = ts;
        if (max_ts > (int)start_ts.size()) start_ts.resize(max_ts);
        for (auto car_i : I) {
            assert(Sch[car_i].start >= ts);
            n++;
            start_ts[Sch[car_i].start].push_back(car_i);
            if (mp.cars[car_i].prior) n_p++;
            if (mp.cars[car_i].preset) n_pre++;
            if (mp.cars[car_i].prior && mp.cars[car_i].preset) n_pre_p++;
        }
        for (; ts < max_ts; ts++) {
            int r = run_step(Sch);
            if (r == FINISH)
                te = int(mp.a_ * tpri + ts + 0.5), tesum = int(mp.b_ * tsumpri + tsum + 0.5);
            if (r != OK) return r;
        }
        return TIMEUP;
    }

    // 查询t时刻道路上的信息: 车辆数，平均车速，平均时差
    std::tuple<int, double, double> get_info(int road_i, bool is_forward, int t) {
        const auto &V = road_info[is_forward][road_i];
        auto p = std::upper_bound(V.begin(), V.end(),
                                  std::make_tuple(t, std::numeric_limits<int>::max(),
                                                  std::numeric_limits<double>::max(),
                                                  std::numeric_limits<double>::max()));
        if (p == V.begin()) return std::make_tuple(0, 0.0, 0.0);
        --p;
        return std::make_tuple(std::get<1>(*p), std::get<2>(*p), std::get<3>(*p));
    };
};
#endif /* ifndef CHECKER_H */
