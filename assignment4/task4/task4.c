/*Задание 4
Реализуйте распределённую программу с использованием MPI для обработки массива
данных. Разделите массив между процессами, выполните вычисления локально и
соберите результаты. Проведите замеры времени выполнения для 2, 4 и 8 процессов.*/

/*limits.h нужен для UINT_MAX*/
#include <limits.h>
/*mpi.h подключает интерфейс MPI*/
#include <mpi.h>
/*stdio.h для printf*/
#include <stdio.h>
/*stdlib.h для malloc/free*/
#include <stdlib.h>

/*простая генерация псевдослучайного числа в диапазоне [0,1]
state-состояние генератора*/
double frand(unsigned int *state) {
    /*линейный конгруэнтный генератор*/
    *state = 1664525u * (*state) + 1013904223u;
    /*нормируем к [0,1]*/
    return (double)(*state) / (double)UINT_MAX;
}

/*главная функция программы*/
int main(int argc, char** argv) {
    /*инициализация MPI*/
    MPI_Init(&argc, &argv);

    /*rank-номер процесса, size-количество процессов*/
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*размер массива*/
    const int N = 10000000; /*10 млн*/

    /*базовое количество элементов на процесс*/
    int local_n = N / size;
    /*остаток, если N не делится на size*/
    int remainder = N % size;

    /*распределяем остаток по первым процессам: первые remainder процессов получают на 1 элемент больше*/
    int my_n = local_n + (rank < remainder ? 1 : 0);

    /*counts и displs нужны для MPI_Scatterv:
  counts[i]-сколько элементов отправить процессу i
  displs[i]-смещение (с какого индекса массива начинать отправку)*/
    int *counts = NULL, *displs = NULL;
    if (rank == 0) {
        counts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));

        int offset = 0;
        for (int i = 0; i < size; i++) {
            counts[i] = local_n + (i < remainder ? 1 : 0);
            displs[i] = offset;
            offset += counts[i];
        }
    }

    /*исходный массив создаём только на процессе 0*/
    double *data = NULL;
    if (rank == 0) {
        data = (double*)malloc((size_t)N * sizeof(double));
        unsigned int st = 123u; /*фиксированный seed*/
        for (int i = 0; i < N; i++) data[i] = frand(&st);
    }

    /*локальный буфер для своей части массива*/
    double *local = (double*)malloc((size_t)my_n * sizeof(double));

    /*барьер чтобы начать замер одновременно*/
    MPI_Barrier(MPI_COMM_WORLD);
    /*MPI_Wtime возвращает время в секундах*/
    double t0 = MPI_Wtime();

    /*раздаём данные по процессам*/
    MPI_Scatterv(data, counts, displs, MPI_DOUBLE,
                 local, my_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /*локальная работа, считаем сумму своего куска*/
    double local_sum = 0.0;
    for (int i = 0; i < my_n; i++) local_sum += local[i];

    /*собираем итоговую сумму на процессе 0*/
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /*барьер перед окончанием замера*/
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    /*вывод результатов делает только процесс 0*/
    if (rank == 0) {
        printf("N=%d, processes=%d\n", N, size);
        printf("Global sum = %.6f\n", global_sum);
        printf("Time (s) = %.6f\n", t1 - t0);
    }

    /*освобождаем локальную память*/
    free(local);

    /*процесс 0 освобождает свои данные и служебные массивы*/
    if (rank == 0) {
        free(data);
        free(counts);
        free(displs);
    }

    /*завершение MPI*/
    MPI_Finalize();
    return 0;
}
