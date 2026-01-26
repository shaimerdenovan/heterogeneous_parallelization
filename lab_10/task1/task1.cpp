/*Задание 1. Анализ производительности CPU-параллельной программы (OpenMP) 
Разработайте параллельную программу на C++ с использованием OpenMP для обработки 
большого массива данных (например, вычисление суммы, среднего значения и 
дисперсии). 
Требуется: 
реализовать базовую параллельную версию; 
выполнить профилирование программы с использованием omp_get_wtime() и/или 
профилировщика (Intel VTune, gprof); 
определить: 
долю параллельной и последовательной части программы; 
влияние числа потоков на ускорение; 
проанализировать результаты в контексте закона Амдала.*/

/*подключаем OpenMP (omp_get_wtime, управление потоками)*/
#include <omp.h>
/*printf*/
#include <cstdio>
/*atoll, atoi*/
#include <cstdlib>
/*математика*/
#include <cmath>
/*vector для хранения большого массива*/
#include <vector>
/*remove_if для фильтра списка потоков*/
#include <algorithm>

/*удобная функция текущее время в секундах*/
static inline double now() { return omp_get_wtime(); }

/*главная функция программы*/
int main(int argc, char** argv) {
    /*N-размер массива, repeats-сколько раз повторяем замер для усреднения*/
    long long N = (argc > 1) ? atoll(argv[1]) : 100000000LL; /*по умолчанию 1e8*/
    int repeats = (argc > 2) ? atoi(argv[2]) : 3;            /*по умолчанию 3 прогона*/

    printf("N=%lld, repeats=%d\n", N, repeats);
    printf("Max threads available: %d\n\n", omp_get_max_threads());

    /*создаём массив на CPU*/
    std::vector<double> a(N);

    /*последовательная часть: заполняем массив*/
    double t0 = now();
    for (long long i = 0; i < N; i++) {
        a[i] = (i % 1000) * 0.001 + 1.0;
    }
    double t_init = now() - t0;

    /*считаем сумму, среднее и дисперсию в одном потоке*/
    auto run_serial = [&]() {
        double t = now();

        /*сумма*/
        double sum = 0.0;
        for (long long i = 0; i < N; i++) sum += a[i];

        /*среднее*/
        double mean = sum / (double)N;

        /*дисперсия*/
        double var = 0.0;
        for (long long i = 0; i < N; i++) {
            double d = a[i] - mean;
            var += d * d;
        }
        var /= (double)N;

        /*время выполнения*/
        double dt = now() - t;

        /*печатаем чтобы компилятор не выкинул вычисления как ненужные*/
        printf("[serial] sum=%.6f mean=%.6f var=%.6f time=%.6f s\n", sum, mean, var, dt);
        return dt;
    };

    /*параллельная версия: sum и var считаем через reduction*/
    auto run_omp_basic = [&]() {
        double t = now();

        /*параллельно считаем сумму (reduction аккуратно сложит частичные суммы потоков)*/
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (long long i = 0; i < N; i++) sum += a[i];

        /*среднее*/
        double mean = sum / (double)N;

        /*параллельно считаем сумму квадратов отклонений для дисперсии*/
        double var_sum = 0.0;
        #pragma omp parallel for reduction(+:var_sum)
        for (long long i = 0; i < N; i++) {
            double d = a[i] - mean;
            var_sum += d * d;
        }

        /*дисперсия*/
        double var = var_sum / (double)N;

        /*время выполнения*/
        double dt = now() - t;

        printf("[omp basic] sum=%.6f mean=%.6f var=%.6f time=%.6f s\n", sum, mean, var, dt);
        return dt;
    };

    /*показываем, сколько заняло заполнение массива*/
    printf("Init time (serial part): %.6f s\n\n", t_init);

    /*меряем последовательную версию несколько раз и берём среднее*/
    double t_serial = 0.0;
    for (int r = 0; r < repeats; r++) t_serial += run_serial();
    t_serial /= repeats;
    printf("\nAverage serial compute time: %.6f s\n\n", t_serial);

    /*список потоков для теста*/
    std::vector<int> thread_list = {1, 2, 4, 8, 16, 32};
    thread_list.erase(
        std::remove_if(thread_list.begin(), thread_list.end(),
                       [&](int x){ return x > omp_get_max_threads(); }),
        thread_list.end()
    );

    /*табличка: потоки, время, ускорение, оценка p (параллельной доли) по Амдалу*/
    printf("Threads, avg_time(s), speedup_vs_serial, est_parallel_fraction(Amdahl)\n");

    for (int th : thread_list) {
        /*ставим число потоков*/
        omp_set_num_threads(th);

        /*меряем параллельную версию несколько раз*/
        double t_par = 0.0;
        for (int r = 0; r < repeats; r++) t_par += run_omp_basic();
        t_par /= repeats;

        /*ускорение относительно последовательного compute*/
        double speedup = t_serial / t_par;

        /*оценка параллельной доли p по закону Амдала*/
        double p = 0.0;
        if (th > 1 && speedup > 0.0) {
            p = (1.0 - 1.0/speedup) / (1.0 - 1.0/(double)th);
        }

        printf("%d, %.6f, %.3f, %.4f\n\n", th, t_par, speedup, p);
    }

    /*завершение программы*/
    return 0;
}
