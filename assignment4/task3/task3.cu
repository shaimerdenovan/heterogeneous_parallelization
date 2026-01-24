/*Задание 3
Реализуйте гибридную программу, в которой обработка массива выполняется
параллельно на CPU и GPU. Первую часть массива обработайте на CPU, вторую — на
GPU. Сравните время выполнения CPU-, GPU- и гибридной реализаций.*/

/*подключаем CUDA Runtime API*/
#include <cuda_runtime.h>
/*библиотека для вывода в консоль*/
#include <iostream>
/*библиотека vector для хранения данных на CPU*/
#include <vector>
/*библиотека random для генерации случайных чисел*/
#include <random>
/*библиотека chrono для замера времени*/
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

/*CUDA-ядро, поэлементная обработка массива
считаем y[i]=a[i]*2+1, offset нужен, чтобы обрабатывать не с нуля, а с середины массива для гибрида*/
__global__ void transform(const float* a, float* y, int n, int offset) {
    /*индекс внутри запуска*/
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    /*глобальный индекс в исходном массиве*/
    int gid = offset + i;

    /*проверяем границы*/
    if (gid < n) y[gid] = a[gid] * 2.0f + 1.0f;
}

/*функция CPU: обработка элементов на отрезке [l, r) и суммирование результата
возвращаем сумму, чтобы было удобно сравнивать корректность*/
double cpu_transform_sum(const std::vector<float>& a, int l, int r) {
    double s = 0.0;
    for (int i = l; i < r; i++) {
        float y = a[i] * 2.0f + 1.0f;
        s += y;
    }
    return s;
}

/*главная функция*/
int main() {
    /*размер массива*/
    const int N = 10'000'000;

    /*массив на CPU*/
    std::vector<float> a(N);

    /*генерация входных данных*/
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; i++) a[i] = dist(rng);

    /*CPU-only, вся обработка выполняется на CPU*/
    auto c0 = std::chrono::high_resolution_clock::now();
    double s_cpu = cpu_transform_sum(a, 0, N);
    auto c1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();

    /*GPU-only, вся обработка выполняется на GPU*/
    float *d_a=nullptr, *d_y=nullptr;

    /*выделяем память на GPU*/
    CUDA_CHECK(cudaMalloc(&d_a, N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, N*sizeof(float)));

    /*копируем входной массив на GPU*/
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), N*sizeof(float), cudaMemcpyHostToDevice));

    /*параметры запуска CUDA*/
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    /*события для замера времени GPU*/
    cudaEvent_t g0,g1;
    CUDA_CHECK(cudaEventCreate(&g0));
    CUDA_CHECK(cudaEventCreate(&g1));

    /*старт замера*/
    CUDA_CHECK(cudaEventRecord(g0));

    /*запуск ядра для всей длины массива*/
    transform<<<blocks, threads>>>(d_a, d_y, N, 0);
    CUDA_CHECK(cudaGetLastError());

    /*конец замера*/
    CUDA_CHECK(cudaEventRecord(g1));
    CUDA_CHECK(cudaEventSynchronize(g1));

    /*время выполнения ядра на GPU*/
    float gpu_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, g0, g1));

    /*для проверки корректности копируем результат на CPU и суммируем*/
    std::vector<float> y_gpu(N);
    CUDA_CHECK(cudaMemcpy(y_gpu.data(), d_y, N*sizeof(float), cudaMemcpyDeviceToHost));

    double s_gpu = 0.0;
    for (int i = 0; i < N; i++) s_gpu += y_gpu[i];

    /*Гибрид, первая часть обрабатывается на CPU, вторая часть на GPU*/
    int mid = N / 2;

    /*GPU запускаем только на второй половине массива*/
    int n2 = N - mid;
    int blocks2 = (n2 + threads - 1) / threads;

    cudaEvent_t h0,h1;
    CUDA_CHECK(cudaEventCreate(&h0));
    CUDA_CHECK(cudaEventCreate(&h1));

    /*CPU часть: считаем от 0 до mid*/
    auto hc0 = std::chrono::high_resolution_clock::now();
    double s_h_cpu = cpu_transform_sum(a, 0, mid);
    auto hc1 = std::chrono::high_resolution_clock::now();
    double hcpu_ms = std::chrono::duration<double, std::milli>(hc1 - hc0).count();

    /*GPU часть: считаем от mid до N*/
    CUDA_CHECK(cudaEventRecord(h0));

    transform<<<blocks2, threads>>>(d_a, d_y, N, mid);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(h1));
    CUDA_CHECK(cudaEventSynchronize(h1));

    float hgpu_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&hgpu_ms, h0, h1));

    /*копируем только вторую половину результата*/
    std::vector<float> y2(n2);
    CUDA_CHECK(cudaMemcpy(y2.data(), d_y + mid, n2*sizeof(float), cudaMemcpyDeviceToHost));

    double s_h_gpu = 0.0;
    for (int i = 0; i < n2; i++) s_h_gpu += y2[i];

    /*общая сумма гибридного варианта*/
    double s_hybrid = s_h_cpu + s_h_gpu;

    /*вывод результатов*/
    std::cout << "N=" << N << "\n";
    std::cout << "CPU-only: time(ms)=" << cpu_ms << ", sum=" << s_cpu << "\n";
    std::cout << "GPU-only(kernel): time(ms)=" << gpu_ms
              << ", sum(after copy+cpu sum)=" << s_gpu << "\n";
    std::cout << "Hybrid: CPU part(ms)=" << hcpu_ms
              << ", GPU kernel part(ms)=" << hgpu_ms
              << ", total(sum)=" << s_hybrid << "\n";
    std::cout << "Diff(CPU-GPU) = " << std::abs(s_cpu - s_gpu) << "\n";
    std::cout << "Diff(CPU-Hybrid) = " << std::abs(s_cpu - s_hybrid) << "\n";

    /*освобождаем память GPU*/
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_y));

    /*удаляем события*/
    CUDA_CHECK(cudaEventDestroy(g0));
    CUDA_CHECK(cudaEventDestroy(g1));
    CUDA_CHECK(cudaEventDestroy(h0));
    CUDA_CHECK(cudaEventDestroy(h1));

    return 0;
}
