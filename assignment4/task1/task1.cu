/*Задание 1
Реализуйте CUDA-программу для вычисления суммы элементов массива с
использованием глобальной памяти. Сравните результат и время выполнения с
последовательной реализацией на CPU для массива размером 100 000 элементов.*/

/*подключаем CUDA Runtime API*/
#include <cuda_runtime.h>
/*библиотека для вывода в консоль*/
#include <iostream>
/*библиотека vector для хранения данных на CPU*/
#include <vector>
/*библиотека random для генерации случайных чисел*/
#include <random>
/*библиотека chrono для замера времени на CPU*/
#include <chrono>

/*макрос для проверки ошибок CUDA*/
#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)  \
                  << " at " << __FILE__ << ":" << __LINE__      \
                  << std::endl;                                 \
        std::exit(1);                                           \
    }                                                           \
} while(0)

/*CUDA-ядро, вычисление суммы массива-каждый поток суммирует часть элементов,
затем внутри блока выполняется редукция в shared памяти, поток tid=0 добавляет сумму блока в глобальную сумму через atomicAdd,
результат хранится в глобальной памяти out.*/
__global__ void sum_atomic_kernel(const float* a, float* out, int n) {
    /*локальная сумма потока*/
    float local = 0.0f;

    /*grid-stride loop: i-глобальный индекс элемента, шаг = blockDim.x*gridDim.x*/
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += blockDim.x * gridDim.x)
    {
        /*добавляем элемент массива в локальную сумму*/
        local += a[i];
    }

    /*редукция в пределах блока, выделяется динамически при запуске ядра: threads*sizeof(float)*/
    extern __shared__ float s[];

    /*локальный индекс потока в блоке*/
    int tid = threadIdx.x;

    /*каждый поток записывает свою локальную сумму в shared*/
    s[tid] = local;

    /*синхронизация, все потоки должны записать данные в shared*/
    __syncthreads();

    /*параллельная редукция деревом внутри блока*/
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        /*половина потоков суммирует пары элементов*/
        if (tid < stride) s[tid] += s[tid + stride];
        /*синхронизация после каждого шага редукции*/
        __syncthreads();
    }

    /*после редукции s[0]-сумма блока, atomicAdd добавляет её к общей сумме в глобальной памяти*/
    if (tid == 0) atomicAdd(out, s[0]);
}

/*последовательное вычисление суммы на CPU*/
float cpu_sum(const std::vector<float>& a) {
    double s = 0.0;
    for (float x : a) s += x;
    return (float)s;
}

/*главная функция программы*/
int main() {
    /*размер массива по условию задания*/
    const int N = 100000;

    /*массив на CPU*/
    std::vector<float> h(N);

    /*генерируем данные*/
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; i++) h[i] = dist(rng);

    /*CPU замер времени*/
    const int ITERS = 200;

    float s_cpu = 0.0f;

    /*старт замера времени на CPU*/
    auto c0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; it++) s_cpu = cpu_sum(h);
    /*конец замера времени на CPU*/
    auto c1 = std::chrono::high_resolution_clock::now();

    /*среднее время одного запуска на CPU (в миллисекундах)*/
    double cpu_ms =
        std::chrono::duration<double, std::milli>(c1 - c0).count() / ITERS;

        /*GPU подготовка данных*/
    float *d_in=nullptr, *d_out=nullptr;

    /*выделяем память на GPU: d_in-массив входных данных,d_out-одна переменная под итоговую сумму*/
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    /*копируем входной массив на GPU*/
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    /*настройка запуска CUDA*/
    int threads = 256; /*потоков в блоке*/
    int blocks = 120;

    /*GPU прогрев*/
    /*обнуляем сумму на GPU*/
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

    /*прогревочный запуск ядра (исключаем влияние первого запуска)*/
    sum_atomic_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /*GPU замер времени*/
    /*cudaEvent измеряет время на стороне GPU*/
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    /*старт события*/
    CUDA_CHECK(cudaEventRecord(e0));

    float s_gpu = 0.0f;

    /*многократный запуск ядра для усреднения*/
    for (int it = 0; it < ITERS; it++) {
        /*обнуляем глобальную сумму перед каждым запуском*/
        CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(float)));

        /*запускаем ядро, разделяемая память=threads*sizeof(float)*/
        sum_atomic_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_out, N);
        CUDA_CHECK(cudaGetLastError());
    }

    /*конечное событие и ожидание завершения*/
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));

    /*общее время (ms) за ITERS запусков*/
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, e0, e1));

    /*среднее время одного запуска на GPU*/
    float gpu_ms = total_ms / ITERS;

    /*копируем сумму обратно на CPU*/
    CUDA_CHECK(cudaMemcpy(&s_gpu, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    /*вывод результатов*/
    std::cout << "N=" << N << "\n";
    std::cout << "CPU avg time(ms)=" << cpu_ms << ", sum=" << s_cpu << "\n";
    std::cout << "GPU avg time(ms)=" << gpu_ms << ", sum=" << s_gpu << "\n";
    std::cout << "Abs diff=" << std::abs(s_cpu - s_gpu) << "\n";

    /*освобождаем память на GPU*/
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    /*удаляем события*/
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    return 0;
}
