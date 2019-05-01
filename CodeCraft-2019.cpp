#define RUN
#include "map.hpp"
#include "checker.hpp"
#include "scheduler.hpp"

#include <memory>
#include <numeric>
#include <iostream>
#include <stdexcept>


int main(int argc, char *argv[]) {
    std::cout << "Begin" << std::endl;
    if (argc < 6) {
        std::cout << "please input args: carPath, roadPath, crossPath, presetAnswerPath, answerPath" <<
                  std::endl;
        exit(1);
    }
    std::string carPath(argv[1]);
    std::string roadPath(argv[2]);
    std::string crossPath(argv[3]);
    std::string presetAnswerPath(argv[4]);
    std::string answerPath(argv[5]);
    std::cout << "carPath is " << carPath << std::endl;
    std::cout << "roadPath is " << roadPath << std::endl;
    std::cout << "crossPath is " << crossPath << std::endl;
    std::cout << "presetAnswerPath is " << presetAnswerPath << std::endl;
    std::cout << "answerPath is " << answerPath << std::endl;
    auto mp = std::unique_ptr<Map>(new Map(carPath, roadPath, crossPath, presetAnswerPath));
    mp->print();
    std::vector<int> I(mp->n_cars);
    iota(I.begin(), I.end(), 0);

#ifdef CHECK
    auto Sch = mp->read_answer(answerPath);
    for (int i = 0; i < mp->n_cars; i++)
        if (mp->cars[i].preset && Sch[i].Path.empty())
            Sch[i] = mp->Preset[i];
    int ttt = clock();
    std::unique_ptr<Checker> ch(new Checker(*mp));
    int s = ch->run(Sch, I, 100000);
    printf(s == DEADLOCK ? "DEADLOCK\n" : s == FINISH ? "FINISH\n" : "TIMEUP\n");
    printf("判题器用时: %.1f\n", double(clock() - ttt) * 1000 / CLOCKS_PER_SEC);
    ch->print();
#endif
#ifdef RUN
    std::vector<Schedule> Sch(mp->n_cars);
    // 读入预置路径
    for (int i = 0; i < mp->n_cars; i++) if (mp->cars[i].preset) Sch[i] = mp->Preset[i];

    int max_ts = 100000, preset_ts, preset_tpri, preset_te;
    std::tie(preset_ts, preset_tpri, preset_te) =
        std::unique_ptr<Scheduler>(new Scheduler(*mp))->change_preset(Sch, 100000, 0.1);

    auto ser = std::unique_ptr<Scheduler>(new Scheduler(*mp));
    int ts, tpri, te;
    std::tie(ts, tpri, te) = ser->gen_schedule(Sch, -1, max_ts, max_ts * (1 + mp->a_), true);
    printf("Margin -1: %d, %d, %d\n", ts, tpri, te);
    mp->print_schedule(Sch, answerPath); 

    int best_te = te;
    int l = 0, r = tpri - preset_tpri;
    while (l + 1 < r) {
        if (double(clock() - start_time) / CLOCKS_PER_SEC > MAX_PROGRAM_SECS) break;
        int margin = l + (r - l + 1) / 3;
        printf("Margin: %d checking\n", margin);
        if (margin + preset_tpri >= tpri) { puts("Margin too large"), r = margin; continue; }
        try {
            std::tie(ts, tpri, te) = ser->gen_schedule(Sch, margin, ts * (1 + mp->a_), best_te, false);
            printf("Margin %d: %d, %d, %d\n", margin, ts, tpri, te);
            if (te < best_te) {
                best_te = te;
                mp->print_schedule(Sch, answerPath);
            }
            r = margin;
        } catch (std::logic_error &e) {
            std::cout << "Margin " << margin << ": " << e.what() << std::endl;
            l = margin;
        }
    }
    printf("Best Te: %d\n", best_te);
#endif
    printf("程序运行用时: %.1f\n", double(clock() - start_time) * 1000 / CLOCKS_PER_SEC);
    return 0;
}
