/*Задание 3. Профилирование гибридного приложения CPU + GPU 
Разработайте гибридную программу, в которой часть вычислений выполняется на CPU, а 
часть — на GPU. 
Требуется: 
1. реализовать гибридный алгоритм обработки массива данных; 
2. использовать асинхронную передачу данных (cudaMemcpyAsync) и CUDA streams; 
3. выполнить профилирование приложения: 
a. определить накладные расходы передачи данных; 
b. выявить узкие места при взаимодействии CPU и GPU; 
4. предложить и реализовать одну оптимизацию, уменьшающую накладные расходы.*/
/*Подключаемые библиотеки*/

/*CUDA Runtime API-работа с памятью GPU, ядрами, потоками и событиями*/
#include <cuda_runtime.h>

/*библиотека для вывода результатов в консоль*/
#include <cstdio>

/*стандартная библиотека C: malloc/free, atoi, exit*/
#include <cstdlib>

/*std::min-используется для определения размера обрабатываемого чанка*/
#include <algorithm>


/*макрос для проверки ошибок CUDA-функций*/
#define CHECK(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err__)); \
    exit(1); \
  } \
} while(0)

/*GPU ядро: нагружаем вычисления чтобы было что считать на видеокарте*/
__global__ void gpu_compute(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        /*несколько итераций чтобы ядро не было слишком пустым*/
        #pragma unroll 8
        for (int k = 0; k < 64; k++) x = x * 1.000001f + 0.000001f;
        out[i] = x;
    }
}

/*CPU часть: простая редукция (суммируем результаты после GPU)*/
double cpu_reduce(const float* a, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i];
    return s;
}

/*главная функция программы*/
int main(int argc, char** argv) {
    /*n-общий размер массива
      chunk-размер одного чанка (сколько элементов обрабатываем за раз)
      nstreams-сколько потоков (streams) используем*/
    int n = (argc > 1) ? atoi(argv[1]) : (1<<26);
    int chunk = (argc > 2) ? atoi(argv[2]) : (1<<22);
    int nstreams = (argc > 3) ? atoi(argv[3]) : 4;

    printf("n=%d chunk=%d streams=%d\n", n, chunk, nstreams);

    /*выделяем pinned память на CPU*/
    float *h_in=nullptr, *h_out=nullptr;
    CHECK(cudaHostAlloc(&h_in,  (size_t)n*sizeof(float), cudaHostAllocDefault));
    CHECK(cudaHostAlloc(&h_out, (size_t)n*sizeof(float), cudaHostAllocDefault));

    /*заполняем входные данные*/
    for (int i = 0; i < n; i++) h_in[i] = (float)(i % 1000) * 0.001f;

    /*создаём streams и буферы на GPU (по одному набору буферов на stream)*/
    cudaStream_t* streams = (cudaStream_t*)malloc(nstreams*sizeof(cudaStream_t));
    float **d_in  = (float**)malloc(nstreams*sizeof(float*));
    float **d_out = (float**)malloc(nstreams*sizeof(float*));

    for (int s = 0; s < nstreams; s++) {
        CHECK(cudaStreamCreate(&streams[s]));
        CHECK(cudaMalloc(&d_in[s],  (size_t)chunk*sizeof(float)));
        CHECK(cudaMalloc(&d_out[s], (size_t)chunk*sizeof(float)));
    }

    /*CUDA события для измерения времени: общее время, отдельно H2D, Kernel, D2H*/
    cudaEvent_t ev_start, ev_stop, ev_h2d_s, ev_h2d_e, ev_k_s, ev_k_e, ev_d2h_s, ev_d2h_e;
    CHECK(cudaEventCreate(&ev_start));
    CHECK(cudaEventCreate(&ev_stop));
    CHECK(cudaEventCreate(&ev_h2d_s));
    CHECK(cudaEventCreate(&ev_h2d_e));
    CHECK(cudaEventCreate(&ev_k_s));
    CHECK(cudaEventCreate(&ev_k_e));
    CHECK(cudaEventCreate(&ev_d2h_s));
    CHECK(cudaEventCreate(&ev_d2h_e));

    int block = 256;

    /*накапливаем времена: суммируем по чанкам*/
    float h2d_ms=0.0f, k_ms=0.0f, d2h_ms=0.0f;

    /*старт общего замера*/
    CHECK(cudaEventRecord(ev_start));

    int offset = 0;  /*с какой позиции массива берём чанк*/
    int iter = 0;    /*номер чанка*/

    while (offset < n) {
        /*выбираем stream по кругу*/
        int s = iter % nstreams;

        /*сколько элементов в текущем чанке*/
        int cur = std::min(chunk, n - offset);

        /*H2D: асинхронно копируем чанк на GPU*/
        CHECK(cudaEventRecord(ev_h2d_s, streams[s]));
        CHECK(cudaMemcpyAsync(d_in[s], h_in + offset, (size_t)cur*sizeof(float),
                              cudaMemcpyHostToDevice, streams[s]));
        CHECK(cudaEventRecord(ev_h2d_e, streams[s]));

        /*Kernel: запускаем вычисление в этом же stream*/
        int grid = (cur + block - 1) / block;
        CHECK(cudaEventRecord(ev_k_s, streams[s]));
        gpu_compute<<<grid, block, 0, streams[s]>>>(d_in[s], d_out[s], cur);
        CHECK(cudaEventRecord(ev_k_e, streams[s]));

        /*D2H: асинхронно копируем результат обратно на CPU*/
        CHECK(cudaEventRecord(ev_d2h_s, streams[s]));
        CHECK(cudaMemcpyAsync(h_out + offset, d_out[s], (size_t)cur*sizeof(float),
                              cudaMemcpyDeviceToHost, streams[s]));
        CHECK(cudaEventRecord(ev_d2h_e, streams[s]));

        /*ждём завершения именно этого stream чтобы можно было корректно снять времена событий*/
        CHECK(cudaStreamSynchronize(streams[s]));

        /*считаем сколько заняли этапы для этого чанка и добавляем в общий счётчик*/
        float ms=0.0f;
        CHECK(cudaEventElapsedTime(&ms, ev_h2d_s, ev_h2d_e)); h2d_ms += ms;
        CHECK(cudaEventElapsedTime(&ms, ev_k_s,   ev_k_e));   k_ms   += ms;
        CHECK(cudaEventElapsedTime(&ms, ev_d2h_s, ev_d2h_e)); d2h_ms += ms;

        offset += cur;
        iter++;
    }

    /*конец общего замера*/
    CHECK(cudaEventRecord(ev_stop));
    CHECK(cudaEventSynchronize(ev_stop));

    float total_ms=0.0f;
    CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_stop));

    /*CPU часть: после того как все данные вернулись, суммируем результат*/
    double cpu_s = cpu_reduce(h_out, n);

    printf("Approx H2D(ms)=%.3f  Kernel(ms)=%.3f  D2H(ms)=%.3f\n", h2d_ms, k_ms, d2h_ms);
    printf("Total wall-time (GPU pipeline + sync) = %.3f ms\n", total_ms);
    printf("CPU reduce sum = %.6f\n", cpu_s);

    /*освобождаем ресурсы*/
    for (int s = 0; s < nstreams; s++) {
        CHECK(cudaFree(d_in[s]));
        CHECK(cudaFree(d_out[s]));
        CHECK(cudaStreamDestroy(streams[s]));
    }
    free(streams);
    free(d_in);
    free(d_out);

    CHECK(cudaFreeHost(h_in));
    CHECK(cudaFreeHost(h_out));

    CHECK(cudaEventDestroy(ev_start));
    CHECK(cudaEventDestroy(ev_stop));
    CHECK(cudaEventDestroy(ev_h2d_s));
    CHECK(cudaEventDestroy(ev_h2d_e));
    CHECK(cudaEventDestroy(ev_k_s));
    CHECK(cudaEventDestroy(ev_k_e));
    CHECK(cudaEventDestroy(ev_d2h_s));
    CHECK(cudaEventDestroy(ev_d2h_e));

    /*завершение программы*/
    return 0;
}
