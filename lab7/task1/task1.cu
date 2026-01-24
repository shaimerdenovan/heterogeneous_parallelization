/*Практическая работа №7
Задание 1: Реализация редукции 
1. Напишите ядро CUDA для выполнения редукции (суммирования 
элементов массива). 
2. Используйте разделяемую память для оптимизации доступа к данным. 
3. Проверьте корректность работы на тестовом массиве.*/

#include <cuda_runtime.h>   /*CUDA Runtime API: cudaMalloc, cudaMemcpy, cudaFree, cudaGetLastError*/
#include <iostream>         /*вывод в консоль через cout*/
#include <vector>           /*контейнер vector для массива на CPU*/
#include <random>           /*генератор случайных чисел*/
#include <cmath>            /*fabs для проверки разницы*/

/*макрос для проверки ошибок CUDA*/
#define CUDA_CHECK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  std::cerr<<"CUDA error: "<<cudaGetErrorString(e)<<" at "<<__FILE__<<":"<<__LINE__<<"\n"; exit(1);} }while(0)

/*CUDA-ядро редукции суммы*/
template<int BLOCK>
__global__ void reduce_sum_kernel(const float* __restrict__ in, float* __restrict__ out, int n){
    /*разделяемая память, быстрый буфер внутри блока, здесь храним частичные суммы для редукции*/
    __shared__ float sdata[BLOCK];

    /*локальный индекс потока внутри блока*/
    unsigned tid = threadIdx.x;

    /*каждый блок обрабатывает 2*BLOCK элементов, i-глобальный индекс первого элемента для данного потока*/
    unsigned i = blockIdx.x * (BLOCK * 2) + tid;

    /*каждый поток читает до 2 элементов из глобальной памяти и складывает их в локальную переменную x*/
    float x = 0.0f;
    /*проверяем границы массива*/
    if (i < (unsigned)n) x += in[i];
    /*второй элемент на расстоянии BLOCK*/
    if (i + BLOCK < (unsigned)n) x += in[i + BLOCK];

    /*записываем локальную сумму потока в shared memory*/
    sdata[tid] = x;

    /*синхронизация, все потоки должны записать sdata*/
    __syncthreads();

    /*параллельная редукция в shared memory, на каждом шаге расстояние (stride) уменьшаем в 2 раза*/
    for (unsigned s = BLOCK/2; s > 0; s >>= 1){
        /*половина потоков складывает элементы попарно*/
        if (tid < s) sdata[tid] += sdata[tid + s];
        /*синхронизация между шагами редукции*/
        __syncthreads();
    }

    /*после редукции итоговая сумма блока лежит в sdata[0], поток 0 записывает её в глобальную память*/
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

/*функция запуска редукции на GPU с многопроходной схемой*/
float gpu_reduce_sum(const float* d_in, int n, int blockSize){
    /*threads=число потоков в блоке*/
    int threads = blockSize;

    /*blocks=число блоков в сетке, каждый блок обрабатывает 2*threads элементов*/
    int blocks = (n + threads*2 - 1) / (threads*2);

    /*временные массивы на GPU для частичных результатов, используем два буфера и переключаемся между ними*/
    float *d_tmp1=nullptr, *d_tmp2=nullptr;
    CUDA_CHECK(cudaMalloc(&d_tmp1, blocks*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tmp2, blocks*sizeof(float)));

    /*текущий вход*/
    const float* cur_in = d_in;
    /*текущий выход*/
    float* cur_out = d_tmp1;

    /*текущий размер массива который редуцируем*/
    int cur_n = n;

    /*флаг для переключения буферов tmp1/tmp2*/
    bool toggle = false;

    /*повторяем редукцию до тех пор, пока не останется 1 блок*/
    while(true){
        /*пересчитываем сколько блоков нужно для текущего размера*/
        blocks = (cur_n + threads*2 - 1) / (threads*2);

        /*запускаем ядро с нужным BLOCK (256/512/1024)*/
        if(blockSize==256)      reduce_sum_kernel<256><<<blocks,256>>>(cur_in, cur_out, cur_n);
        else if(blockSize==512) reduce_sum_kernel<512><<<blocks,512>>>(cur_in, cur_out, cur_n);
        else if(blockSize==1024)reduce_sum_kernel<1024><<<blocks,1024>>>(cur_in, cur_out, cur_n);
        else { std::cerr<<"Use blockSize 256/512/1024\n"; exit(1); }

        /*проверяем ошибку запуска ядра*/
        CUDA_CHECK(cudaGetLastError());

        /*если блок всего один - редукция завершена*/
        if(blocks == 1) break;

        /*иначе теперь нужно редуцировать массив частичных сумм*/
        cur_n = blocks;     /*новый размер=число блоков*/
        cur_in = cur_out;   /*новый вход=предыдущий выход*/

        /*переключаем буфер для следующей записи*/
        toggle = !toggle;
        cur_out = toggle ? d_tmp2 : d_tmp1;
    }

    /*копируем итог с GPU на CPU*/
    float res = 0.0f;
    CUDA_CHECK(cudaMemcpy(&res, cur_out, sizeof(float), cudaMemcpyDeviceToHost));

    /*освобождаем временную память GPU*/
    CUDA_CHECK(cudaFree(d_tmp1));
    CUDA_CHECK(cudaFree(d_tmp2));

    /*возвращаем сумму*/
    return res;
}

/*главная функция программы*/
int main(){
    /*размер тестового массива*/
    int n = 1024;

    /*размер блока потоков*/
    int blockSize = 256;

    /*генератор случайных чисел*/
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    /*массив на CPU*/
    std::vector<float> h(n);
    /*заполняем массив случайными значениями*/
    for(int i=0;i<n;i++) h[i] = dist(rng);

    /*сумма на CPU*/
    double cpu = 0.0;
    for(float v : h) cpu += v;

    /*указатель на память GPU для входного массива*/
    float *d_in = nullptr;

    /*выделяем память на GPU*/
    CUDA_CHECK(cudaMalloc(&d_in, n*sizeof(float)));

    /*копируем массив с CPU на GPU*/
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), n*sizeof(float), cudaMemcpyHostToDevice));

    /*вычисляем сумму на GPU через редукцию*/
    float gpu = gpu_reduce_sum(d_in, n, blockSize);

    /*освобождаем память GPU*/
    CUDA_CHECK(cudaFree(d_in));

    /*вывод результатов*/
    std::cout<<"CPU sum = "<<cpu<<"\n";
    std::cout<<"GPU sum = "<<gpu<<"\n";
    std::cout<<"abs diff = "<<std::fabs(cpu - (double)gpu)<<"\n";

    /*проверка корректности (допуск для float)*/
    if(std::fabs(cpu - (double)gpu) < 1e-2) std::cout<<"OK: correct\n";
    else std::cout<<"FAIL: incorrect\n";

    return 0;
}
