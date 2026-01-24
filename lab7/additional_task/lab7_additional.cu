/*Практическая работа №7
Дополнительные задания 
1. Реализуйте редукцию для нахождения минимума и максимума. 
2. Реализуйте алгоритм Blelloch Scan для более эффективного 
сканирования. 
3. Исследуйте влияние размера блока на производительность.*/

#include <cuda_runtime.h>   /*CUDA Runtime API*/
#include <iostream>         /*вывод в консоль*/
#include <vector>           /*vector для массивов на CPU*/
#include <random>           /*случайные числа*/
#include <algorithm>        /*minmax_element, max*/
#include <cmath>            /*fabs, fminf, fmaxf*/
#include <limits>           /*numeric_limits*/

/*макрос проверки ошибок CUDA*/
#define CUDA_CHECK(call) do{ cudaError_t _err=(call); if(_err!=cudaSuccess){ \
  std::cerr<<"CUDA error: "<<cudaGetErrorString(_err)<<" at "<<__FILE__<<":"<<__LINE__<<"\n"; std::exit(1);} }while(0)

/*Редукция суммы*/

/*CUDA-ядро редукции суммы для одного прохода*/
template<int BLOCK>
__global__ void reduce_sum_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int n)
{
    /*разделяемая память для частичных сумм*/
    __shared__ float sdata[BLOCK];

    /*локальный индекс потока*/
    unsigned tid = threadIdx.x;

    /*глобальный индекс первого элемента для потока (каждый поток читает 2 элемента)*/
    unsigned i = blockIdx.x * (BLOCK * 2) + tid;

    /*локальная сумма потока*/
    float x = 0.0f;

    /*читаем первый элемент*/
    if(i < (unsigned)n) x += in[i];

    /*читаем второй элемент*/
    if(i + BLOCK < (unsigned)n) x += in[i + BLOCK];

    /*пишем локальную сумму в разделяемую память*/
    sdata[tid] = x;

    /*синхронизация потоков блока*/
    __syncthreads();

    /*редукция в shared memory (stride уменьшается в 2 раза)*/
    for(unsigned s = BLOCK/2; s > 0; s >>= 1){
        if(tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    /*поток 0 записывает сумму блока*/
    if(tid == 0) out[blockIdx.x] = sdata[0];
}

/*многопроходная GPU редукция суммы: повторяем редукцию над массивом частичных сумм, пока не останется 1 значение*/
float gpu_reduce_sum(const float* d_in, int n, int blockSize){
    int threads = blockSize;
    int blocks  = (n + threads*2 - 1) / (threads*2);

    /*два временных буфера для частичных сумм*/
    float *t1 = nullptr, *t2 = nullptr;
    CUDA_CHECK(cudaMalloc(&t1, blocks*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&t2, blocks*sizeof(float)));

    const float* cur_in = d_in; /*текущий вход*/
    float* cur_out = t1;        /*текущий выход*/
    int cur_n = n;              /*текущий размер*/
    bool toggle = false;        /*переключение буферов*/

    while(true){
        blocks = (cur_n + threads*2 - 1) / (threads*2);

        /*выбор шаблона по blockSize*/
        if(blockSize==256)       reduce_sum_kernel<256><<<blocks,256>>>(cur_in,cur_out,cur_n);
        else if(blockSize==512)  reduce_sum_kernel<512><<<blocks,512>>>(cur_in,cur_out,cur_n);
        else if(blockSize==1024) reduce_sum_kernel<1024><<<blocks,1024>>>(cur_in,cur_out,cur_n);
        else { std::cerr<<"Use blockSize 256/512/1024\n"; std::exit(1); }

        CUDA_CHECK(cudaGetLastError());

        /*если один блок то готово*/
        if(blocks == 1) break;

        /*подготовка следующего прохода*/
        cur_n = blocks;
        cur_in = cur_out;
        toggle = !toggle;
        cur_out = toggle ? t2 : t1;
    }

    /*копируем итоговую сумму на CPU*/
    float res = 0.0f;
    CUDA_CHECK(cudaMemcpy(&res, cur_out, sizeof(float), cudaMemcpyDeviceToHost));

    /*освобождаем память*/
    CUDA_CHECK(cudaFree(t1));
    CUDA_CHECK(cudaFree(t2));

    return res;
}

/*Редукция min/max*/

/*ядро которое в одном проходе даёт min и max для каждого блока*/
template<int BLOCK>
__global__ void reduce_minmax_kernel(const float* __restrict__ in,
                                     float* __restrict__ out_min,
                                     float* __restrict__ out_max,
                                     int n)
{
    /*разделяемая память для min и max отдельно*/
    __shared__ float smin[BLOCK];
    __shared__ float smax[BLOCK];

    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * (BLOCK * 2) + tid;

    /*инициализация экстремальными значениями*/
    float mn = +INFINITY;
    float mx = -INFINITY;

    /*читаем до 2 элементов и обновляем min/max*/
    if(i < (unsigned)n){
        float v = in[i];
        mn = fminf(mn, v);
        mx = fmaxf(mx, v);
    }
    if(i + BLOCK < (unsigned)n){
        float v = in[i + BLOCK];
        mn = fminf(mn, v);
        mx = fmaxf(mx, v);
    }

    /*записываем локальные min/max в разделяемую память*/
    smin[tid] = mn;
    smax[tid] = mx;
    __syncthreads();

    /*редукция min/max в разделяемую память*/
    for(unsigned s = BLOCK/2; s > 0; s >>= 1){
        if(tid < s){
            smin[tid] = fminf(smin[tid], smin[tid + s]);
            smax[tid] = fmaxf(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }

    /*поток 0 записывает min и max блока*/
    if(tid == 0){
        out_min[blockIdx.x] = smin[0];
        out_max[blockIdx.x] = smax[0];
    }
}

/*ядро редукции только min*/
template<int BLOCK>
__global__ void reduce_min_kernel(const float* __restrict__ in, float* __restrict__ out, int n){
    __shared__ float s[BLOCK];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * (BLOCK * 2) + tid;

    float mn = +INFINITY;
    if(i < (unsigned)n) mn = fminf(mn, in[i]);
    if(i + BLOCK < (unsigned)n) mn = fminf(mn, in[i + BLOCK]);

    s[tid] = mn;
    __syncthreads();

    for(unsigned step = BLOCK/2; step > 0; step >>= 1){
        if(tid < step) s[tid] = fminf(s[tid], s[tid + step]);
        __syncthreads();
    }

    if(tid==0) out[blockIdx.x] = s[0];
}

/*ядро редукции только max*/
template<int BLOCK>
__global__ void reduce_max_kernel(const float* __restrict__ in, float* __restrict__ out, int n){
    __shared__ float s[BLOCK];
    unsigned tid = threadIdx.x;
    unsigned i = blockIdx.x * (BLOCK * 2) + tid;

    float mx = -INFINITY;
    if(i < (unsigned)n) mx = fmaxf(mx, in[i]);
    if(i + BLOCK < (unsigned)n) mx = fmaxf(mx, in[i + BLOCK]);

    s[tid] = mx;
    __syncthreads();

    for(unsigned step = BLOCK/2; step > 0; step >>= 1){
        if(tid < step) s[tid] = fmaxf(s[tid], s[tid + step]);
        __syncthreads();
    }

    if(tid==0) out[blockIdx.x] = s[0];
}

/*многопроходная редукция минимума для массива d_in размера n*/
float gpu_reduce_min_only(const float* d_in, int n, int blockSize){
    int threads = blockSize;
    int blocks  = (n + threads*2 - 1) / (threads*2);

    float *t1=nullptr, *t2=nullptr;
    CUDA_CHECK(cudaMalloc(&t1, blocks*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&t2, blocks*sizeof(float)));

    const float* cur_in = d_in;
    float* cur_out = t1;
    int cur_n = n;
    bool toggle=false;

    while(true){
        blocks = (cur_n + threads*2 - 1) / (threads*2);

        if(blockSize==256)       reduce_min_kernel<256><<<blocks,256>>>(cur_in,cur_out,cur_n);
        else if(blockSize==512)  reduce_min_kernel<512><<<blocks,512>>>(cur_in,cur_out,cur_n);
        else if(blockSize==1024) reduce_min_kernel<1024><<<blocks,1024>>>(cur_in,cur_out,cur_n);
        else { std::cerr<<"Use blockSize 256/512/1024\n"; std::exit(1); }

        CUDA_CHECK(cudaGetLastError());
        if(blocks==1) break;

        cur_n = blocks;
        cur_in = cur_out;
        toggle = !toggle;
        cur_out = toggle ? t2 : t1;
    }

    float res = 0.0f;
    CUDA_CHECK(cudaMemcpy(&res, cur_out, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(t1));
    CUDA_CHECK(cudaFree(t2));
    return res;
}

/*многопроходная редукция максимума*/
float gpu_reduce_max_only(const float* d_in, int n, int blockSize){
    int threads = blockSize;
    int blocks  = (n + threads*2 - 1) / (threads*2);

    float *t1=nullptr, *t2=nullptr;
    CUDA_CHECK(cudaMalloc(&t1, blocks*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&t2, blocks*sizeof(float)));

    const float* cur_in = d_in;
    float* cur_out = t1;
    int cur_n = n;
    bool toggle=false;

    while(true){
        blocks = (cur_n + threads*2 - 1) / (threads*2);

        if(blockSize==256)       reduce_max_kernel<256><<<blocks,256>>>(cur_in,cur_out,cur_n);
        else if(blockSize==512)  reduce_max_kernel<512><<<blocks,512>>>(cur_in,cur_out,cur_n);
        else if(blockSize==1024) reduce_max_kernel<1024><<<blocks,1024>>>(cur_in,cur_out,cur_n);
        else { std::cerr<<"Use blockSize 256/512/1024\n"; std::exit(1); }

        CUDA_CHECK(cudaGetLastError());
        if(blocks==1) break;

        cur_n = blocks;
        cur_in = cur_out;
        toggle = !toggle;
        cur_out = toggle ? t2 : t1;
    }

    float res = 0.0f;
    CUDA_CHECK(cudaMemcpy(&res, cur_out, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(t1));
    CUDA_CHECK(cudaFree(t2));
    return res;
}

/*Blelloch scan (exclusive)*/

/*CUDA ядро блочного Blelloch exclusive scan (2*BLOCK элементов на блок)*/
template<int BLOCK>
__global__ void block_exclusive_scan(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     float* __restrict__ block_sums,
                                     int n)
{
    /*разделяемая память на 2*BLOCK элементов*/
    __shared__ float temp[2*BLOCK];

    int tid   = threadIdx.x;
    int start = 2 * BLOCK * blockIdx.x;

    int ai = start + tid;
    int bi = start + tid + BLOCK;

    /*загрузка данных в разделяемую память*/
    temp[tid]       = (ai < n) ? in[ai] : 0.0f;
    temp[tid+BLOCK] = (bi < n) ? in[bi] : 0.0f;

    /*Upsweep, строим суммы вверх*/
    int offset = 1;
    for(int d = BLOCK; d > 0; d >>= 1){
        __syncthreads();
        if(tid < d){
            int i1 = offset*(2*tid+1) - 1;
            int i2 = offset*(2*tid+2) - 1;
            temp[i2] += temp[i1];
        }
        offset <<= 1;
    }

    /*сохраняем сумму блока и обнуляем корень для exclusive scan*/
    if(tid == 0){
        if(block_sums) block_sums[blockIdx.x] = temp[2*BLOCK - 1];
        temp[2*BLOCK - 1] = 0.0f;
    }

    /*Downsweep, распространяем суммы вниз*/
    for(int d = 1; d <= BLOCK; d <<= 1){
        offset >>= 1;
        __syncthreads();
        if(tid < d){
            int i1 = offset*(2*tid+1) - 1;
            int i2 = offset*(2*tid+2) - 1;
            float t = temp[i1];
            temp[i1] = temp[i2];
            temp[i2] += t;
        }
    }
    __syncthreads();

    /*запись результата блока*/
    if(ai < n) out[ai] = temp[tid];
    if(bi < n) out[bi] = temp[tid+BLOCK];
}

/*ядро добавления смещений для каждого блока*/
__global__ void add_block_offsets(float* data,
                                  const float* offsets,
                                  int n,
                                  int elemsPerBlock)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n){
        int b = i / elemsPerBlock;
        data[i] += offsets[b];
    }
}

/*функция scan на GPU (рекурсивная)*/
void gpu_exclusive_scan(const float* d_in, float* d_out, int n, int blockSize){
    int elemsPerBlock = 2*blockSize;
    int blocks = (n + elemsPerBlock - 1) / elemsPerBlock;

    float *d_sums=nullptr, *d_offsets=nullptr;

    /*если блоков больше одного то выделяем память под суммы и offsets*/
    if(blocks > 1){
        CUDA_CHECK(cudaMalloc(&d_sums, blocks*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_offsets, blocks*sizeof(float)));
    }

    /*scan внутри каждого блока*/
    if(blockSize==256)       block_exclusive_scan<256><<<blocks,256>>>(d_in, d_out, d_sums, n);
    else if(blockSize==512)  block_exclusive_scan<512><<<blocks,512>>>(d_in, d_out, d_sums, n);
    else if(blockSize==1024) block_exclusive_scan<1024><<<blocks,1024>>>(d_in, d_out, d_sums, n);
    else { std::cerr<<"Use blockSize 256/512/1024\n"; std::exit(1); }

    CUDA_CHECK(cudaGetLastError());

    /*если блоков больше одного то рекурсивно сканируем суммы блоков и добавляем offsets*/
    if(blocks > 1){
        gpu_exclusive_scan(d_sums, d_offsets, blocks, blockSize);

        int threads = 256;
        int grid = (n + threads - 1) / threads;
        add_block_offsets<<<grid,threads>>>(d_out, d_offsets, n, elemsPerBlock);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(d_sums));
        CUDA_CHECK(cudaFree(d_offsets));
    }
}

/*Функции замера времени через cudaEvent (GPU timing)*/

/*замер времени GPU редукции суммы*/
float time_gpu_reduce_sum(const float* d_in, int n, int blockSize, int iters=10){
    /*события для замера времени на GPU*/
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /*прогрев*/
    gpu_reduce_sum(d_in, n, blockSize);
    CUDA_CHECK(cudaDeviceSynchronize());

    /*старт таймера*/
    CUDA_CHECK(cudaEventRecord(start));

    /*несколько запусков для усреднения*/
    for(int i=0;i<iters;i++) gpu_reduce_sum(d_in, n, blockSize);

    /*стоп таймера*/
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    /*удаляем события*/
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters;
}

/*замер времени GPU scan*/
float time_gpu_scan(const float* d_in, float* d_out, int n, int blockSize, int iters=10){
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /*прогрев*/
    gpu_exclusive_scan(d_in, d_out, n, blockSize);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for(int i=0;i<iters;i++) gpu_exclusive_scan(d_in, d_out, n, blockSize);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters;
}

/*Функция main*/
int main(){
    /*генератор случайных чисел*/
    std::mt19937 rng(123);

    /*распределение для scan/редукции суммы*/
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    /*распределение для min/max*/
    std::uniform_real_distribution<float> distMM(-100.0f, 100.0f);

    /*Тест редукции min/max*/
    {
        int n = 1'000'000;
        int blockSize = 256;

        /*создаём массив на CPU*/
        std::vector<float> h(n);
        for(int i=0;i<n;i++) h[i] = distMM(rng);

        /*CPU эталон (minmax_element)*/
        auto [itMin, itMax] = std::minmax_element(h.begin(), h.end());
        float cpuMin = *itMin;
        float cpuMax = *itMax;

        /*копируем массив на GPU*/
        float *d_in = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, n*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h.data(), n*sizeof(float), cudaMemcpyHostToDevice));

        /*число блоков для первого прохода*/
        int blocks = (n + blockSize*2 - 1) / (blockSize*2);

        /*массивы частичных min и max по блокам*/
        float *d_part_min = nullptr, *d_part_max = nullptr;
        CUDA_CHECK(cudaMalloc(&d_part_min, blocks*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_part_max, blocks*sizeof(float)));

        /*первый проход, считаем min/max на каждый блок*/
        reduce_minmax_kernel<256><<<blocks,256>>>(d_in, d_part_min, d_part_max, n);
        CUDA_CHECK(cudaGetLastError());

        /*второй этап, редуцируем массивы частичных min/max до одного значения*/
        float gpuMin = gpu_reduce_min_only(d_part_min, blocks, blockSize);
        float gpuMax = gpu_reduce_max_only(d_part_max, blocks, blockSize);

        /*освобождаем память*/
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_part_min));
        CUDA_CHECK(cudaFree(d_part_max));

        /*вывод результатов*/
        std::cout<<"Min/Max reduction test\n";
        std::cout<<"CPU min/max: "<<cpuMin<<"  "<<cpuMax<<"\n";
        std::cout<<"GPU min/max: "<<gpuMin<<"  "<<gpuMax<<"\n";
        std::cout<<"abs diff min: "<<std::fabs(cpuMin-gpuMin)<<"\n";
        std::cout<<"abs diff max: "<<std::fabs(cpuMax-gpuMax)<<"\n";
        std::cout<<((std::fabs(cpuMin-gpuMin)<1e-4 && std::fabs(cpuMax-gpuMax)<1e-4) ? "OK\n\n" : "FAIL\n\n");
    }

    /*Тест корректности Blelloch scan на большом массиве*/
    {
        int n = 1'000'000;
        int blockSize = 256;

        /*входной массив*/
        std::vector<float> h(n);
        for(int i=0;i<n;i++) h[i] = dist01(rng);

        /*копируем на GPU*/
        float *d_in = nullptr, *d_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, n*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, n*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, h.data(), n*sizeof(float), cudaMemcpyHostToDevice));

        /*GPU scan*/
        gpu_exclusive_scan(d_in, d_out, n, blockSize);

        /*копируем результат на CPU*/
        std::vector<float> gpu(n);
        CUDA_CHECK(cudaMemcpy(gpu.data(), d_out, n*sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));

        /*CPU exclusive scan*/
        std::vector<double> cpu(n);
        double acc = 0.0;
        for(int i=0;i<n;i++){
            cpu[i] = acc;
            acc += (double)h[i];
        }

        /*считаем максимальную абсолютную и относительную ошибку*/
        double max_abs = 0.0;
        double max_rel = 0.0;
        int worst_i = -1;

        for(int i=0;i<n;i++){
            double c = cpu[i];
            double g = (double)gpu[i];
            double abs_err = std::fabs(c - g);
            if(abs_err > max_abs){ max_abs = abs_err; worst_i = i; }
            double denom = std::max(1.0, std::fabs(c));
            max_rel = std::max(max_rel, abs_err / denom);
        }

        std::cout<<"Blelloch exclusive scan test\n";
        std::cout<<"Max abs error: "<<max_abs<<"\n";
        std::cout<<"Max rel error: "<<max_rel<<"\n";
        if(worst_i >= 0){
            std::cout<<"Worst index: "<<worst_i<<" cpu="<<cpu[worst_i]<<" gpu="<<gpu[worst_i]<<"\n";
        }
        std::cout<<"First 5 values (cpu vs gpu):\n";
        for(int i=0;i<5;i++) std::cout<<i<<": "<<cpu[i]<<"  "<<gpu[i]<<"\n";

        /*допускиЮ для float на больших n нормальна небольшая abs ошибка*/
        bool ok = (max_rel < 1e-5) || (max_abs < 1e-1);
        std::cout<<(ok ? "OK\n\n" : "FAIL\n\n");
    }

    /*Замеры влияния размера блока на производительность (GPU редукция суммы и GPU scan)*/
    {
        std::vector<int> sizes  = {1024, 1'000'000, 10'000'000};
        std::vector<int> blocks = {256, 512, 1024};

        std::cout<<"Block size performance CSV\n";
        std::cout<<"n,block,gpu_reduce_sum_ms,gpu_blelloch_scan_ms\n";

        for(int n : sizes){
            /*генерируем входные данные*/
            std::vector<float> h(n);
            for(int i=0;i<n;i++) h[i] = dist01(rng);

            /*копируем на GPU*/
            float *d_in = nullptr, *d_out = nullptr;
            CUDA_CHECK(cudaMalloc(&d_in, n*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_out, n*sizeof(float)));
            CUDA_CHECK(cudaMemcpy(d_in, h.data(), n*sizeof(float), cudaMemcpyHostToDevice));

            /*замеры для разных blockSize*/
            for(int b : blocks){
                float red = time_gpu_reduce_sum(d_in, n, b, 10);
                float scn = time_gpu_scan(d_in, d_out, n, b, 10);
                std::cout<<n<<","<<b<<","<<red<<","<<scn<<"\n";
            }

            CUDA_CHECK(cudaFree(d_in));
            CUDA_CHECK(cudaFree(d_out));
        }
    }

    return 0;
}
