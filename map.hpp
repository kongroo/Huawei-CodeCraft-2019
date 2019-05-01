#ifndef MAP_H
#define MAP_H

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>


enum { LEFT, RIGHT, STRAIGHT, FIN };
struct Car { int i, id, from, to, speed, start, prior = 0, preset = 0; };
struct Road { int i, id, len, limit, num, from, to, bi; };
struct Cross { int i, id, roads[4]; };
struct Schedule { int start, change; std::vector<std::pair<int, double>> Path; };

struct Map {
    // 用0~n_cars/n_roads/n_crosses的idx表示车辆，道路，路口
    int n_cars, n_roads, n_crosses;
    std::unordered_map<int, int> cross_idx, car_idx, road_idx; // 原始id向idx表示转化
    std::vector<Car> cars;
    std::vector<Road> roads;
    std::vector<Cross> crosses;
    std::vector<std::vector<std::pair<int, int>>> G; // 地图
    std::vector<std::vector<int>> InG; // 每个路口的道路
    std::vector<std::array<int, 4>> Turn[4]; // 道路转向后的道路, Turn[INDEX][CROSS][DIR]
    std::vector<std::vector<char>> RI; // 道路在路口处的index
    std::vector<Schedule> Preset;

    // 地图统计量
    double a_, b_; // 加权系数
    int n_prior = 0, n_preset = 0, n_pp = 0; // 各种车辆数
    int mav = -1, mav_prior = -1, miv = 1e9, miv_prior = 1e9; // 车速
    int late = -1, early = 1e9, late_prior = -1,
        late_preset = -1, late_pp = -1, early_prior = 1e9; // 发车时间
    int n_from, n_to, n_from_p, n_to_p; // 出发到达点分布
    std::vector<int> cross_nlanes, cross_nroads; // 路口车道数，道路数

    Map(const std::string &car_path, const std::string &road_path,
        const std::string &cross_path, const std::string &preset_path) {
        load(car_path, road_path, cross_path, preset_path);
    }

    // 载入地图
    void load(const std::string &car_path, const std::string &road_path,
              const std::string &cross_path, const std::string &preset_path) {
        // 初始化
        n_cars = n_roads = n_crosses = 0;
        cross_idx.clear(), car_idx.clear(), road_idx.clear();
        cars.clear(), roads.clear(), crosses.clear();
        G.clear(), InG.clear();
        Preset.clear();
        std::set<int> From, To, PFrom, PTo;
        std::string s;
        // 读入车辆文件
        std::ifstream in_car(car_path);
        while (getline(in_car, s)) {
            if (s.front() == '(') {
                Car c;
#ifdef NOPRESET
                sscanf(s.c_str(), "(%d,%d,%d,%d,%d)", &c.id, &c.from,
                       &c.to, &c.speed, &c.start);
#else
                sscanf(s.c_str(), "(%d,%d,%d,%d,%d,%d,%d)", &c.id, &c.from,
                       &c.to, &c.speed, &c.start, &c.prior, &c.preset);
#endif
                mav = std::max(mav, c.speed), miv = std::min(miv, c.speed);
                late = std::max(late, c.start), early = std::min(early, c.start);
                From.insert(c.from), To.insert(c.to);
                if (c.prior) {
                    n_prior++;
                    mav_prior = std::max(mav_prior, c.speed), miv_prior = std::min(miv_prior, c.speed);
                    late_prior = std::max(late_prior, c.start), early_prior = std::min(early_prior, c.start);
                    PFrom.insert(c.from), PTo.insert(c.to);
                }
                if (c.preset) late_preset = std::max(late_preset, c.start), n_preset++;
                if (c.prior && c.preset) late_pp = std::max(late_pp, c.start), n_pp++;
                cars.emplace_back(c), n_cars++;
            }
        }
        n_from = From.size(), n_to = To.size(), n_from_p = PFrom.size(), n_to_p = PTo.size();
        sort(cars.begin(), cars.end(), [](const Car & a, const Car & b) { return a.id < b.id; });
        for (int i = 0; i < n_cars; i++) car_idx[cars[i].id] = cars[i].i = i;
        Preset.resize(n_cars);
        // 读入路口文件
        std::ifstream in_cross(cross_path);
        while (getline(in_cross, s)) {
            if (s.front() == '(') {
                Cross c;
                sscanf(s.c_str(), "(%d,%d,%d,%d,%d)", &c.id, &c.roads[0], &c.roads[1], &c.roads[2], &c.roads[3]);
                crosses.emplace_back(c), n_crosses++;
            }
        }
        sort(crosses.begin(), crosses.end(), [](const Cross & a, const Cross & b) { return a.id < b.id; });
        for (int i = 0; i < n_crosses; i++) cross_idx[crosses[i].id] = crosses[i].i = i;
        G.resize(n_crosses), InG.resize(n_crosses);
        // 读入道路文件
        std::ifstream in_road(road_path);
        while (getline(in_road, s)) {
            if (s.front() == '(') {
                Road r;
                sscanf(s.c_str(), "(%d,%d,%d,%d,%d,%d,%d)",
                       &r.id, &r.len, &r.limit, &r.num, &r.from, &r.to, &r.bi);
                roads.emplace_back(r), n_roads++;
            }
        }
        sort(roads.begin(), roads.end(), [](const Road & a, const Road & b) { return a.id < b.id; });
        for (int i = 0; i < n_roads; i++) {
            auto &r = roads[i];
            int u = cross_idx[r.from], v = cross_idx[r.to];
            G[u].emplace_back(v, i), InG[v].push_back(i);
            if (r.bi) G[v].emplace_back(u, i), InG[u].push_back(i);
            road_idx[r.id] = r.i = i;
        }
        // 读入预置车辆路线
#ifndef NOPRESET
        std::ifstream in_preset(preset_path);
        while (getline(in_preset, s)) {
            if (s.front() == '(') {
                for (int i = 0; i < (int)s.size(); i++) if (!isdigit(s[i])) s[i] = ' ';
                std::stringstream ss(s);
                int id, car_i, road_id;
                ss >> id, car_i = car_idx[id];
                ss >> Preset[car_i].start;
                late_preset = std::max(late_preset, Preset[car_i].start);
                if (cars[car_i].prior) late_pp = std::max(late_pp, Preset[car_i].start);
                while (ss >> road_id) Preset[car_i].Path.emplace_back(road_id, 0.0);
            }
        }
#endif
        // 将数据结构中的原始id转化成idx表示
        for (auto &car : cars) {
            car.from = cross_idx[car.from];
            car.to = cross_idx[car.to];
        }
        for (auto &road : roads) {
            road.from = cross_idx[road.from];
            road.to = cross_idx[road.to];
        }
        for (auto &cross : crosses) {
            for (int i = 0; i < 4; i++) {
                if (cross.roads[i] != -1)
                    cross.roads[i] = road_idx[cross.roads[i]];
            }
        }
        for (int i = 0; i < n_cars; i++) if (cars[i].preset)
                for (auto &r : Preset[i].Path) r.first = road_idx[r.first];
        // 计算系数
        auto round5 = [](double x) { return int(x * 100000 + 0.5) / 100000.; };
        double x = round5(n_cars / double(std::max(1, n_prior)));
        double y = round5(round5(mav / double(miv)) / round5(mav_prior / double(miv_prior)));
        double z = round5(round5(late / double(early)) / round5(late_prior / double(early_prior)));
        double w = round5(n_from / double(std::max(1, n_from_p)));
        double v = round5(n_to / double(std::max(1, n_to_p)));
        a_ = 0.05 * x + 0.2375 * (y + z + w + v);
        b_ = 0.8 * x + 0.05 * (y + z + w + v);
        if (!n_prior) a_ = b_ = 0.0;

        // 预处理路口和道路的关系
        for (int i = 0; i < 4; i++) {
            Turn[i].resize(n_crosses);
            for (int j = 0; j < n_crosses; j++)
                for (int k = 0; k < 4; k++)
                    Turn[i][j][k] = -1;
        }
        RI.assign(n_crosses, std::vector<char>(n_roads, -1));
        auto get_road = [&](int road_i, int cross_i, int dir) {
            const auto &croads = crosses[cross_i].roads;
            int i = std::find(croads, croads + 4, road_i) - croads;
            i += (dir == RIGHT) ? 3 : dir == LEFT ? 1 : 2, i %= 4;
            return croads[i];
        };
        cross_nlanes.assign(n_crosses, 0), cross_nroads.assign(n_crosses, 0);
        for (int cross_i = 0; cross_i < n_crosses; cross_i++) {
            for (int i = 0; i < 4; i++) {
                int road_i = crosses[cross_i].roads[i];
                if (road_i == -1) continue;
                cross_nlanes[cross_i] += roads[road_i].num * (roads[road_i].bi + 1);
                cross_nroads[cross_i]++;
                RI[cross_i][road_i] = i;
                Turn[i][cross_i][RIGHT] = get_road(road_i, cross_i, RIGHT);
                Turn[i][cross_i][LEFT] = get_road(road_i, cross_i, LEFT);
                Turn[i][cross_i][STRAIGHT] = get_road(road_i, cross_i, STRAIGHT);
            }
        }
        // 去除无法达到的车速
        int max_limit = 0;
        for (int i = 0; i < n_roads; i++) {
            roads[i].limit = std::min(roads[i].limit, mav);
            max_limit = std::max(roads[i].limit, max_limit);
        }
        for (int i = 0; i < n_cars; i++) cars[i].speed = std::min(cars[i].speed, max_limit);
    }

    void print() {
        printf("总道路数: %d\n", n_roads);
        printf("总路口数: %d\n", n_crosses);
        printf("车辆数: %d (%d), 预设: %d (%d)\n", n_cars, n_prior, n_preset, n_pp);
        printf("最高车速: %d (%d)\n", mav, mav_prior);
        printf("最低车速: %d (%d)\n", miv, miv_prior);
        printf("最早发车: %d (%d)\n", early, early_prior);
        printf("最晚发车: %d (%d)\n", late, late_prior);
        printf("预设最晚发车: %d (%d)\n", late_preset, late_pp);
        printf("出发分布: %d (%d)\n", n_from, n_from_p);
        printf("到达分布: %d (%d)\n", n_to, n_to_p);
        printf("a: %.6f, b: %.6f\n", a_, b_);
    }

    std::vector<Schedule> read_answer(const std::string &ans_path) {
        std::string s;
        std::ifstream ans(ans_path);
        std::vector<Schedule> Sch(n_cars);
        while (getline(ans, s)) {
            if (s.front() == '(') {
                for (int i = 0; i < (int)s.size(); i++) if (!isdigit(s[i])) s[i] = ' ';
                std::stringstream ss(s);
                int id, car_i, road_id;
                ss >> id, car_i = car_idx[id];
                ss >> Sch[car_i].start;
                while (ss >> road_id) Sch[car_i].Path.emplace_back(road_idx[road_id], 0.0);
            }
        }
        return Sch;
    }

    void print_schedule(const std::vector<Schedule> &Sch, const std::string &filename = "") {
        assert((int)Sch.size() == n_cars);
        std::stringstream ss;
        for (int i = 0; i < n_cars; i++) {
            if (!cars[i].preset || Sch[i].change) {
                ss << "(" << cars[i].id << "," << Sch[i].start;
                for (auto r : Sch[i].Path) ss << "," <<  roads[r.first].id;
                ss << ")\n";
            }
        }
        std::string s = ss.str();
        auto h = std::hash<std::string>()(s);
        std::cout << "Answer Hash Value: " << h << std::endl;
        if (filename.empty()) std::cout << s;
        else std::ofstream(filename) << s;
    }
};

#endif /* ifndef MAP_H */
