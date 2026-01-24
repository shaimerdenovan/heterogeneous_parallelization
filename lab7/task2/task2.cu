/*Практическая работа №7
Задание 2: Реализация префиксной суммы 
1. Напишите ядро CUDA для выполнения префиксной суммы. 
2. Используйте разделяемую память для оптимизации доступа к данным. 
3. Проверьте корректность работы на тестовом массиве.*/

#include <cuda_runtime.h>   /*CUDA Runtime API*/
#include <iostream>         /*вывод в консоль*/
#include <vector>           /*контейнер vector для массивов на CPU*/
#include <random>           /*генератор случайных чисел*/
#include <cmath>            /*fabs для проверки ошибок*/
#include <algorithm>        /*std::max*/

/*макрос для проверки ошибок CUDA*/
#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" at "<<__FILE__<<":"<<__LINE__<<"\n"; exit(1);} }while(0)

/*CUDA-ядро блочного Blelloch exclusive scan*/
template<int BLOCK>
__global__ void block_exclusive_scan(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     float* __restrict__ block_sums,
                                     int n)
{
    /*разделяемая память для текущего блока, используется для upsweep и downsweep фаз*/
    __shared__ float temp[2*BLOCK];

    /*локальный индекс потока*/
    int tid = threadIdx.x;

    /*начальный индекс блока в глобальном массиве*/
    int start = 2 * BLOCK * blockIdx.x;

    /*индексы двух элементов обрабатываемых потоком*/
    int ai = start + tid;
    int bi = start + tid + BLOCK;

    /*загрузка данных из глобальной памяти в shared, если индекс выходит за пределы массива то записываем 0*/
    temp[tid]       = (ai < n) ? in[ai] : 0.0f;
    temp[tid+BLOCK] = (bi < n) ? in[bi] : 0.0f;

    /*Фаза Upsweep (редукция),накапливаем суммы в виде бинарного дерева*/
    int offset = 1;
    for(int d = BLOCK; d > 0; d >>= 1){
        /*ждём пока все потоки обновят temp*/
        __syncthreads();
        if(tid < d){
            int i1 = offset * (2*tid + 1) - 1;
            int i2 = offset * (2*tid + 2) - 1;
            temp[i2] += temp[i1];
        }
        offset <<= 1;
    }

    /*после upsweep, последний элемент содержит сумму всего блока, сохраняем её в block_sums,
обнуляем корень для exclusive scan*/
    if(tid == 0){
        if(block_sums) block_sums[blockIdx.x] = temp[2*BLOCK - 1];
        temp[2*BLOCK - 1] = 0.0f;
    }

    /*Фаза Downsweep, распространяем частичные суммы вниз по дереву*/
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

    /*записываем результат из разделяемой памяти обратно в глобальную память*/
    if(ai < n) out[ai] = temp[tid];
    if(bi < n) out[bi] = temp[tid + BLOCK];
}

/*CUDA-ядро добавления смещений, каждый поток добавляет соответствующее смещение своего блока*/
__global__ void add_block_offsets(float* data,
                                  const float* block_offsets,
                                  int n,
                                  int elemsPerBlock)
{
    /*глобальный индекс элемента*/
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        /*номер блока к которому относится элемент*/
        int b = i / elemsPerBlock;
        data[i] += block_offsets[b];
    }
}

/*функция GPU exclusive scan, реализует рекурсивный Blelloch scan для массивов произвольного размера*/
void gpu_exclusive_scan(const float* d_in, float* d_out, int n, int blockSize){
    /*число элементов обрабатываемых одним блоком*/
    int elemsPerBlock = 2 * blockSize;

    /*число блоков*/
    int blocks = (n + elemsPerBlock - 1) / elemsPerBlock;

    /*массивы для сумм блоков и их сканированных значений*/
    float *d_block_sums = nullptr;
    float *d_block_offsets = nullptr;

    /*выделяем память если блоков больше одного*/
    if(blocks > 1){
        CUDA_CHECK(cudaMalloc(&d_block_sums, blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_block_offsets, blocks * sizeof(float)));
    }

    /*запуск CUDA-ядра блочного scan*/
    if(blockSize == 256)      block_exclusive_scan<256><<<blocks,256>>>(d_in,d_out,d_block_sums,n);
    else if(blockSize == 512) block_exclusive_scan<512><<<blocks,512>>>(d_in,d_out,d_block_sums,n);
    else if(blockSize == 1024)block_exclusive_scan<1024><<<blocks,1024>>>(d_in,d_out,d_block_sums,n);
    else { std::cerr<<"Use blockSize 256/512/1024\n"; exit(1); }

    /*проверяем ошибки запуска*/
    CUDA_CHECK(cudaGetLastError());

    /*если блоков больше одного то нужно просканировать суммы блоков*/
    if(blocks > 1){
        /*рекурсивный вызов scan для массива сумм блоков*/
        gpu_exclusive_scan(d_block_sums, d_block_offsets, blocks, blockSize);

        /*добавление смещений ко всем элементам параллельно*/
        int threads = 256;
        int grid = (n + threads - 1) / threads;
        add_block_offsets<<<grid,threads>>>(d_out, d_block_offsets, n, elemsPerBlock);
        CUDA_CHECK(cudaGetLastError());

        /*освобождаем временную память*/
        CUDA_CHECK(cudaFree(d_block_sums));
        CUDA_CHECK(cudaFree(d_block_offsets));
    }
}

/*главная функция программы*/
int main(){
    /*размер тестового массива*/
    int n = 1024;
    /*размер блока потоков*/
    int blockSize = 256;

    /*генерация входных данных*/
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> h(n);
    for(int i=0;i<n;i++) h[i] = dist(rng);

    /*CPU exclusive scan (эталон)*/
    std::vector<float> cpu(n);
    float acc = 0.0f;
    for(int i=0;i<n;i++){
        cpu[i] = acc;
        acc += h[i];
    }

    /*выделение памяти на GPU*/
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n*sizeof(float)));

    /*копирование данных на GPU*/
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), n*sizeof(float), cudaMemcpyHostToDevice));

    /*запуск GPU exclusive scan*/
    gpu_exclusive_scan(d_in, d_out, n, blockSize);

    /*копирование результата обратно на CPU*/
    std::vector<float> gpu(n);
    CUDA_CHECK(cudaMemcpy(gpu.data(), d_out, n*sizeof(float), cudaMemcpyDeviceToHost));

    /*освобождение памяти GPU*/
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    /*проверка корректности*/
    double max_err = 0.0;
    for(int i=0;i<n;i++){
        max_err = std::max(max_err, (double)std::fabs(cpu[i] - gpu[i]));
    }

    /*вывод результатов*/
    std::cout<<"Max abs error: "<<max_err<<"\n";
    std::cout<<"First 10 values (cpu vs gpu):\n";
    for(int i=0;i<10;i++){
        std::cout<<i<<": "<<cpu[i]<<"  "<<gpu[i]<<"\n";
    }

    if(max_err < 1e-2) std::cout<<"OK: correct\n";
    else std::cout<<"FAIL: incorrect\n";

    return 0;
}
