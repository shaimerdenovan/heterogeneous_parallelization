/*Задание 4. Анализ масштабируемости распределённой программы (MPI) 
Реализуйте распределённую программу на MPI для вычисления агрегатной функции над 
большим массивом (например, сумма, минимум, максимум). 
Требуется: 
измерить время выполнения при различном числе процессов; 
оценить strong scaling и weak scaling; 
проанализировать влияние коммуникационных операций (MPI_Reduce, MPI_Allreduce); 
сделать вывод о масштабируемости алгоритма и его практических ограничениях. */

/*Подключаемые библиотеки*/
/*основная библиотека MPI: инициализация, коммуникации, синхронизация, замер времени*/
#include <mpi.h>

/*библиотека для работы с вводом/выводом в стиле C*/
#include <cstdio>

/*используется для преобразования аргументов командной строки (atoi, atoll)*/
#include <cstdlib>

/*используется для функций std::min и std::max при вычислении минимума и максимума*/
#include <algorithm>

/*используется для хранения локального массива данных на каждом MPI-процессе*/
#include <vector>

/*главная функция программы*/
int main(int argc, char** argv) {
    /*инициализация MPI*/
    MPI_Init(&argc, &argv);

    /*rank-номер процесса, size-общее число процессов*/
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*режимы работы: strong-фиксированный глобальный размер массива, локальный размер уменьшается с ростом числа процессов
      weak-фиксированный локальный размер, глобальный размер растёт с числом процессов*/
    const char* mode = (argc > 1) ? argv[1] : (char*)"strong";

    /*глобальный размер массива (для strong scaling)*/
    long long N_global = (argc > 2) ? atoll(argv[2]) : 200000000LL;

    /*локальный размер массива (для weak scaling)*/
    long long N_local_fixed = (argc > 3) ? atoll(argv[3]) : 50000000LL;

    /*число повторов измерений*/
    int iters = (argc > 4) ? atoi(argv[4]) : 5;

    /*определяем локальный размер массива для каждого процесса*/
    long long N_local = 0;
    if (std::string(mode) == "strong") {
        /*strong scaling: делим глобальный массив между процессами*/
        N_local = N_global / size;
    } else {
        /*weak scaling: каждый процесс обрабатывает одинаковый объём данных*/
        N_local = N_local_fixed;
        N_global = N_local * size;
    }

    /*локальный массив данных каждого процесса*/
    std::vector<double> a(N_local);

    /*заполняем массив: данные немного отличаются между процессами (зависит от rank)*/
    for (long long i = 0; i < N_local; i++) {
        a[i] = 1.0 + 0.000001 * (double)((i + 12345LL * rank) % 1000000);
    }

    /*синхронизация всех процессов перед началом измерений*/
    MPI_Barrier(MPI_COMM_WORLD);

    /*лучшие (минимальные) времена по итерациям*/
    double best_compute = 1e100;
    double best_reduce = 1e100;
    double best_allreduce = 1e100;
    double best_total = 1e100;

    /*основной цикл измерений*/
    for (int r = 0; r < iters; r++) {
        /*синхронизация перед каждой итерацией*/
        MPI_Barrier(MPI_COMM_WORLD);

        double t0 = MPI_Wtime();

        /*Вычисления*/
        double t_c0 = MPI_Wtime();

        double local_sum = 0.0;
        double local_min = a[0];
        double local_max = a[0];

        /*каждый процесс считает сумму, минимум и максимум локально*/
        for (long long i = 0; i < N_local; i++) {
            double x = a[i];
            local_sum += x;
            local_min = std::min(local_min, x);
            local_max = std::max(local_max, x);
        }

        double t_c1 = MPI_Wtime();

        /*Коммуникация: Reduce*/
        double global_sum = 0.0, global_min = 0.0, global_max = 0.0;

        double t_r0 = MPI_Wtime();
        MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        double t_r1 = MPI_Wtime();

        /*Коммуникация: Allreduce*/
        double all_sum = 0.0, all_min = 0.0, all_max = 0.0;

        double t_a0 = MPI_Wtime();
        MPI_Allreduce(&local_sum, &all_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&local_min, &all_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&local_max, &all_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        double t_a1 = MPI_Wtime();

        double t1 = MPI_Wtime();

        /*времена для текущей итерации*/
        double compute = t_c1 - t_c0;
        double red = t_r1 - t_r0;
        double allred = t_a1 - t_a0;
        double total = t1 - t0;

        /*обновляем лучшие (минимальные) времена*/
        best_compute = std::min(best_compute, compute);
        best_reduce = std::min(best_reduce, red);
        best_allreduce = std::min(best_allreduce, allred);
        best_total = std::min(best_total, total);

        /*показываем корректность результата (только один раз, на root)*/
        if (rank == 0 && r == iters - 1) {
            printf("Example Reduce result: sum=%.3f min=%.6f max=%.6f\n",
                   global_sum, global_min, global_max);
            printf("Example Allreduce result: sum=%.3f min=%.6f max=%.6f\n",
                   all_sum, all_min, all_max);
        }
    }

    /*собираем худшие (максимальные) времена среди всех процессов: это показывает реальное время выполнения параллельной программы*/
    double out[4] = {best_compute, best_reduce, best_allreduce, best_total};
    double out_max[4];

    MPI_Reduce(out, out_max, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /*вывод итогов (только root)*/
    if (rank == 0) {
        printf("\nMode=%s, ranks=%d, N_global=%lld, N_local=%lld\n",
               mode, size, N_global, N_local);
        printf("Best (worst-rank) compute:   %.6f s\n", out_max[0]);
        printf("Best (worst-rank) Reduce:    %.6f s\n", out_max[1]);
        printf("Best (worst-rank) Allreduce: %.6f s\n", out_max[2]);
        printf("Best (worst-rank) Total:     %.6f s\n", out_max[3]);
    }

    /*завершаем работу MPI*/
    MPI_Finalize();
    return 0;
}
