/*Задание 2. Оптимизация доступа к памяти на GPU (CUDA) 
Реализуйте ядро CUDA для обработки массива данных, демонстрирующее разные 
паттерны доступа к памяти. 
Требуется: 
1. реализовать две версии ядра: 
a. с эффективным (коалесцированным) доступом к глобальной памяти; 
b. с неэффективным доступом к памяти; 
2. измерить время выполнения с использованием cudaEvent; 
3. провести оптимизацию за счёт: 
a. использования разделяемой памяти; 
b. изменения организации потоков; 
4. сравнить результаты и сделать выводы о влиянии доступа к памяти на 
производительность GPU. */

/*CUDA Runtime API*/
#include <cuda_runtime.h>
/*printf*/
#include <cstdio>
/*malloc/atoi/exit*/
#include <cstdlib>

/*макрос для проверки ошибок CUDA функций*/
#define CHECK(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    exit(1); \
  } \
} while(0)

/*проверяем, что ядро реально запустилось*/
#define CHECK_LAUNCH() do { \
  cudaError_t err__ = cudaGetLastError(); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "KERNEL LAUNCH error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    exit(1); \
  } \
} while(0)

/*ядро с хорошим доступом к памяти: соседние потоки читают соседние элементы, транзакции памяти хорошо склеиваются*/
__global__ void kernel_coalesced(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;
}

/*ядро с плохим доступом к памяти: поток i читает in[(i*stride)%n], 
из-за stride адреса между потоками далеко значит хуже коалесцирование*/
__global__ void kernel_uncoalesced(const float* __restrict__ in, float* __restrict__ out, int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int idx = (int)(((long long)i * (long long)stride) % n);
        out[i] = in[idx] * 2.0f;
    }
}

/*наивный stencil 1D, каждый поток делает 3 чтения из global памяти*/
__global__ void kernel_stencil_naive(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float left  = (i > 0)   ? in[i-1] : 0.0f;
        float mid   = in[i];
        float right = (i+1 < n) ? in[i+1] : 0.0f;
        out[i] = left + mid + right;
    }
}

/*stencil с использованием разделяемой памяти*/
__global__ void kernel_stencil_shared(const float* __restrict__ in, float* __restrict__ out, int n) {
    extern __shared__ float sh[];
    int tid = threadIdx.x;
    int base = blockIdx.x * blockDim.x;
    int i = base + tid;

    /*центральный элемент для каждого потока*/
    sh[tid + 1] = (i < n) ? in[i] : 0.0f;

    /*докидываем halo слева (делает только поток tid=0)*/
    if (tid == 0) {
        int il = base - 1;
        sh[0] = (il >= 0) ? in[il] : 0.0f;
    }

    /*докидываем halo справа (делает последний поток в блоке)*/
    if (tid == blockDim.x - 1) {
        int ir = base + blockDim.x;
        sh[blockDim.x + 1] = (ir < n) ? in[ir] : 0.0f;
    }

    /*ждём пока разделяемая память полностью заполнится*/
    __syncthreads();

    /*считаем stencil из разделяемой памяти*/
    if (i < n) out[i] = sh[tid] + sh[tid + 1] + sh[tid + 2];
}

/*выводим информацию про видеокарту*/
static void print_device() {
    int dev = 0;
    CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, dev));
    printf("GPU: %s, SMs=%d, clock=%.0f MHz, mem=%.1f GB\n",
           p.name, p.multiProcessorCount, p.clockRate/1000.0, p.totalGlobalMem/1024.0/1024.0/1024.0);
}

/*главная функция программы*/
int main(int argc, char** argv) {
    int n      = (argc > 1) ? atoi(argv[1]) : (1<<26);  /*размер массива*/
    int iters  = (argc > 2) ? atoi(argv[2]) : 200;      /*сколько раз гоняем ядро для усреднения*/
    int stride = (argc > 3) ? atoi(argv[3]) : 4096;     /*stride для плохого доступа*/
    int block  = (argc > 4) ? atoi(argv[4]) : 256;      /*потоков в блоке*/

    print_device();
    printf("n=%d, iters=%d, stride=%d, block=%d\n", n, iters, stride, block);

    /*готовим входные данные на CPU*/
    size_t bytes = (size_t)n * sizeof(float);
    float *h_in = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) h_in[i] = (float)(i % 1000) * 0.001f;

    /*выделяем память на GPU*/
    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));

    /*копируем массив на GPU*/
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    /*считаем параметры запуска*/
    int grid = (n + block - 1) / block;

    /*события CUDA для измерения времени на GPU*/
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    /*функция для замеров одного варианта ядра*/
    auto bench = [&](const char* name, double bytes_per_iter, auto launcher) {
        /*прогрев*/
        launcher();
        CHECK_LAUNCH();
        CHECK(cudaDeviceSynchronize());

        /*синхронизация перед стартом измерений*/
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(start, 0));

        /*много запусков подряд*/
        for (int k = 0; k < iters; k++) {
            launcher();
            CHECK_LAUNCH();
        }

        /*конец замера*/
        CHECK(cudaEventRecord(stop, 0));
        CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, start, stop));

        /*примерная пропускная способность для наглядности*/
        double sec = ms / 1000.0;
        double gb = (bytes_per_iter * iters) / 1e9;
        double gbps = gb / sec;

        printf("%s: %.3f ms total, %.3f ms/iter, %.2f GB/s\n",
               name, ms, ms/iters, gbps);
    };

    /*coalesced: 1 чтение+1 запись float=8 байт на элемент*/
    bench("coalesced", 8.0 * n, [&](){
        kernel_coalesced<<<grid, block>>>(d_in, d_out, n);
    });

    /*uncoalesced: те же 8 байт но паттерн доступа к памяти хуже*/
    bench("uncoalesced", 8.0 * n, [&](){
        kernel_uncoalesced<<<grid, block>>>(d_in, d_out, n, stride);
    });

    /*stencil naive*/
    bench("stencil naive(global)", 16.0 * n, [&](){
        kernel_stencil_naive<<<grid, block>>>(d_in, d_out, n);
    });

    /*stencil shared*/
    size_t shmem = (block + 2) * sizeof(float);
    bench("stencil shared", 12.0 * n, [&](){
        kernel_stencil_shared<<<grid, block, shmem>>>(d_in, d_out, n);
    });

    /*проверка где читаем один элемент обратно, чтобы убедиться, что всё реально считалось*/
    float check = 0.0f;
    CHECK(cudaMemcpy(&check, d_out + (n/2), sizeof(float), cudaMemcpyDeviceToHost));
    printf("Sanity d_out[n/2]=%f\n", check);

    /*чистим ресурсы*/
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    free(h_in);

    /*завершение программы*/
    return 0;
}
