/*Практическая работа №7
Задание 3: Анализ производительности 
1. Замерьте время выполнения редукции и сканирования для массивов 
разного размера. 
2. Сравните производительность с CPU-реализацией. 
3. Проведите оптимизацию кода, используя различные типы памяти 
CUDA.*/

#include <cuda_runtime.h>   /*CUDA Runtime API*/
#include <iostream>         /*вывод в консоль*/
#include <vector>           /*vector для массивов на CPU*/
#include <random>           /*случайные числа*/
#include <chrono>           /*таймер для CPU*/
#include <cmath>            /*fabs и др*/

/*макрос проверки ошибок CUDA*/
#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" at "<<__FILE__<<":"<<__LINE__<<"\n"; exit(1);} }while(0)

/*GPU редукция суммы*/

/*CUDA-ядро редукции суммы для блока*/
template<int BLOCK>
__global__ void reduce_sum_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int n)
{
    /*разделяемая память, быстрый буфер внутри одного блока*/
    __shared__ float sdata[BLOCK];

    /*локальный индекс потока*/
    unsigned tid = threadIdx.x;

    /*глобальный индекс элемента для потока, каждый поток читает 2 элемента*/
    unsigned i = blockIdx.x * (BLOCK * 2) + tid;

    /*локальная сумма потока*/
    float x = 0.0f;

    /*чтение первого элемента*/
    if(i < (unsigned)n) x += in[i];

    /*чтение второго элемента*/
    if(i + BLOCK < (unsigned)n) x += in[i + BLOCK];

    /*запись частичной суммы потока в разделяемую память*/
    sdata[tid] = x;

    /*барьер, ждём пока все потоки запишут sdata*/
    __syncthreads();

    /*параллельная редукция в shared memory*/
    for(unsigned s = BLOCK/2; s > 0; s >>= 1){
        if(tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    /*поток 0 записывает сумму блока в глобальную память*/
    if(tid == 0) out[blockIdx.x] = sdata[0];
}

/*GPU редукция суммы для массива произвольного размера*/
float gpu_reduce_sum(const float* d_in, int n, int blockSize){
    /*число потоков в блоке*/
    int threads = blockSize;

    /*число блоков в первом проходе, каждый блок обрабатывает 2*threads элементов*/
    int blocks = (n + threads*2 - 1) / (threads*2);

    /*два временных буфера для частичных сумм*/
    float *t1 = nullptr, *t2 = nullptr;
    CUDA_CHECK(cudaMalloc(&t1, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&t2, blocks * sizeof(float)));

    /*текущий вход и выход*/
    const float* cur_in = d_in;
    float* cur_out = t1;

    /*текущий размер массива который редуцируем*/
    int cur_n = n;

    /*переключатель буферов*/
    bool toggle = false;

    /*повторяем редукцию пока blocks не станет 1*/
    while(true){
        blocks = (cur_n + threads*2 - 1) / (threads*2);

        /*запуск ядра с нужным шаблоном*/
        if(blockSize==256)      reduce_sum_kernel<256><<<blocks,256>>>(cur_in,cur_out,cur_n);
        else if(blockSize==512) reduce_sum_kernel<512><<<blocks,512>>>(cur_in,cur_out,cur_n);
        else if(blockSize==1024)reduce_sum_kernel<1024><<<blocks,1024>>>(cur_in,cur_out,cur_n);
        else { std::cerr<<"Use blockSize 256/512/1024\n"; exit(1); }

        /*проверка ошибок запуска*/
        CUDA_CHECK(cudaGetLastError());

        /*если остался один блок то сумма готова*/
        if(blocks == 1) break;

        /*иначе редуцируем массив частичных сумм*/
        cur_n = blocks;
        cur_in = cur_out;

        /*меняем буфер назначения*/
        toggle = !toggle;
        cur_out = toggle ? t2 : t1;
    }

    /*копируем итоговую сумму на CPU*/
    float res = 0.0f;
    CUDA_CHECK(cudaMemcpy(&res, cur_out, sizeof(float), cudaMemcpyDeviceToHost));

    /*освобождаем временные буферы*/
    CUDA_CHECK(cudaFree(t1));
    CUDA_CHECK(cudaFree(t2));

    return res;
}

/*GPU scan (exclusive prefix sum)-Blelloch scan*/

/*CUDA-ядро блочного Blelloch exclusive scan*/
template<int BLOCK>
__global__ void block_exclusive_scan(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     float* __restrict__ block_sums,
                                     int n)
{
    /*разделяемая память на 2*BLOCK элементов*/
    __shared__ float temp[2*BLOCK];

    int tid = threadIdx.x;
    int start = 2 * BLOCK * blockIdx.x;

    /*индексы двух элементов которые обрабатывает поток*/
    int ai = start + tid;
    int bi = start + tid + BLOCK;

    /*загрузка в разделяемую память*/
    temp[tid]       = (ai < n) ? in[ai] : 0.0f;
    temp[tid+BLOCK] = (bi < n) ? in[bi] : 0.0f;

    /*Upsweep (редукция вверх по дереву)*/
    int offset = 1;
    for(int d = BLOCK; d > 0; d >>= 1){
        __syncthreads();
        if(tid < d){
            int i1 = offset * (2*tid + 1) - 1;
            int i2 = offset * (2*tid + 2) - 1;
            temp[i2] += temp[i1];
        }
        offset <<= 1;
    }

    /*сохраняем сумму блока и обнуляем корень для exclusive scan*/
    if(tid == 0){
        if(block_sums) block_sums[blockIdx.x] = temp[2*BLOCK - 1];
        temp[2*BLOCK - 1] = 0.0f;
    }

    /*Downsweep (распространение вниз)*/
    for(int d = 1; d <= BLOCK; d <<= 1){
        offset >>= 1;
        __syncthreads();
        if(tid < d){
            int i1 = offset * (2*tid + 1) - 1;
            int i2 = offset * (2*tid + 2) - 1;
            float t = temp[i1];
            temp[i1] = temp[i2];
            temp[i2] += t;
        }
    }

    __syncthreads();

    /*запись результата обратно в глобальную память*/
    if(ai < n) out[ai] = temp[tid];
    if(bi < n) out[bi] = temp[tid+BLOCK];
}

/*CUDA-ядро добавления смещений*/
__global__ void add_block_offsets(float* data,
                                  const float* offsets,
                                  int n,
                                  int elemsPerBlock)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        int b = i / elemsPerBlock;   /*номер блока для этого элемента*/
        data[i] += offsets[b];
    }
}

/*GPU exclusive scan для массива произвольного размера:
делаем scan внутри блоков, если блоков больше одного то сканируем суммы блоков (рекурсивно), добавляем смещения*/
void gpu_exclusive_scan(const float* d_in, float* d_out, int n, int blockSize){
    int elemsPerBlock = 2 * blockSize;
    int blocks = (n + elemsPerBlock - 1) / elemsPerBlock;

    float *d_sums = nullptr;
    float *d_offsets = nullptr;

    /*если блоков больше одного то нужны массивы сумм и смещений*/
    if(blocks > 1){
        CUDA_CHECK(cudaMalloc(&d_sums, blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_offsets, blocks * sizeof(float)));
    }

    /*запуск блочного scan*/
    if(blockSize==256)      block_exclusive_scan<256><<<blocks,256>>>(d_in,d_out,d_sums,n);
    else if(blockSize==512) block_exclusive_scan<512><<<blocks,512>>>(d_in,d_out,d_sums,n);
    else if(blockSize==1024)block_exclusive_scan<1024><<<blocks,1024>>>(d_in,d_out,d_sums,n);
    else { std::cerr<<"Use blockSize 256/512/1024\n"; exit(1); }

    CUDA_CHECK(cudaGetLastError());

    /*если блоков больше одного то считаем offsets*/
    if(blocks > 1){
        /*рекурсивный scan сумм блоков*/
        gpu_exclusive_scan(d_sums, d_offsets, blocks, blockSize);

        /*параллельно добавляем offsets ко всем элементам*/
        int threads = 256;
        int grid = (n + threads - 1) / threads;
        add_block_offsets<<<grid,threads>>>(d_out, d_offsets, n, elemsPerBlock);
        CUDA_CHECK(cudaGetLastError());

        /*освобождаем временную память*/
        CUDA_CHECK(cudaFree(d_sums));
        CUDA_CHECK(cudaFree(d_offsets));
    }
}

/*CPU реализации*/

/*CPU сумма массива*/
double cpu_sum(const std::vector<float>& a){
    double s = 0.0;
    for(float v : a) s += v;
    return s;
}

/*CPU exclusive scan*/
void cpu_exclusive_scan(const std::vector<float>& in, std::vector<float>& out){
    float acc = 0.0f;
    for(size_t i = 0; i < in.size(); i++){
        out[i] = acc;
        acc += in[i];
    }
}

/*Замеры времени GPU через cudaEvent*/

/*замер времени GPU-редукции*/
float time_gpu_reduce(const float* d_in, int n, int blockSize, int iters=10){
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /*прогрев чтобы исключить влияние первого запуска*/
    gpu_reduce_sum(d_in, n, blockSize);
    CUDA_CHECK(cudaDeviceSynchronize());

    /*старт события*/
    CUDA_CHECK(cudaEventRecord(start));

    /*несколько запусков для среднего времени*/
    for(int i=0;i<iters;i++){
        gpu_reduce_sum(d_in, n, blockSize);
    }

    /*конечное событие*/
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    /*время в миллисекундах*/
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    /*удаляем события*/
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters;
}

/*замер времени GPU-scan*/
float time_gpu_scan(const float* d_in, float* d_out, int n, int blockSize, int iters=10){
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /*прогрев*/
    gpu_exclusive_scan(d_in, d_out, n, blockSize);
    CUDA_CHECK(cudaDeviceSynchronize());

    /*старт*/
    CUDA_CHECK(cudaEventRecord(start));

    /*повторы*/
    for(int i=0;i<iters;i++){
        gpu_exclusive_scan(d_in, d_out, n, blockSize);
    }

    /*стоп*/
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters;
}

/*Главная функция: прогон для разных N и blockSize*/
int main(){
    /*размеры массивов по заданию*/
    std::vector<int> sizes  = {1024, 1000000, 10000000};

    /*разные размеры блока для сравнения*/
    std::vector<int> blocks = {256, 512, 1024};

    /*генерация случайных чисел*/
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    /*печать заголовка таблицы CSV*/
    std::cout<<"n,block, cpu_reduce_ms, gpu_reduce_ms, cpu_scan_ms, gpu_scan_ms\n";

    /*цикл по разным размерам массива*/
    for(int n : sizes){
        /*создаём входной массив на CPU*/
        std::vector<float> h(n);
        for(int i=0;i<n;i++) h[i] = dist(rng);

        /*выделяем память на GPU*/
        float *d_in = nullptr, *d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

        /*копируем входные данные на GPU*/
        CUDA_CHECK(cudaMemcpy(d_in, h.data(), n*sizeof(float), cudaMemcpyHostToDevice));

        /*буфер под CPU scan*/
        std::vector<float> cpu_scan_out(n);

        /*цикл по разным размерам блока GPU*/
        for(int b : blocks){
            /*CPU редукция (время)*/
            auto t0 = std::chrono::high_resolution_clock::now();
            /*volatile чтобы компилятор не выкинул вычисление*/
            volatile double cs = cpu_sum(h);
            auto t1 = std::chrono::high_resolution_clock::now();
            double cpu_red_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();

            /*GPU редукция (время)*/
            float gpu_red_ms = time_gpu_reduce(d_in, n, b, 10);

            /*CPU scan (время)*/
            t0 = std::chrono::high_resolution_clock::now();
            cpu_exclusive_scan(h, cpu_scan_out);
            t1 = std::chrono::high_resolution_clock::now();
            double cpu_scan_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();

            /*GPU scan (время)*/
            float gpu_scan_ms = time_gpu_scan(d_in, d_out, n, b, 10);

            /*вывод строки CSV*/
            std::cout<<n<<","<<b<<","<<cpu_red_ms<<","<<gpu_red_ms<<","<<cpu_scan_ms<<","<<gpu_scan_ms<<"\n";
        }

        /*освобождаем память GPU*/
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
    }

    return 0;
}
