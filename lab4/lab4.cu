/*Assignment 4. Архитектура GPU и оптимизация CUDA-программ
1) Подготовка данных:
● Реализовать программу для генерации массива случайных чисел
(размер: 1,000,000 элементов).
2) Оптимизация параллельного редукционного алгоритма:
● Реализовать редукцию суммы элементов массива с
использованием:
a. Только глобальной памяти.
b. Комбинации глобальной и разделяемой памяти.
● Сравнить производительность и объяснить влияние использования
разделяемой памяти.
3) Оптимизация сортировки на GPU:
● Реализовать сортировку пузырьком для небольших подмассивов с
использованием локальной памяти.
● Использовать глобальную память для хранения общего массива.
● Реализовать слияние отсортированных подмассивов с использованием
разделяемой памяти.
4) Измерение производительности:
● Замерить время выполнения программ с использованием разных типов
памяти для массивов размером 10,000, 100,000 и 1,000,000 элементов.
● Построить графики зависимости времени выполнения от размера
массива*/

/*подключаем библиотеку CUDA Runtime API*/
#include <cuda_runtime.h>
/*библиотека для вывода printf/fprintf*/
#include <cstdio>
/*библиотека для exit*/
#include <cstdlib>
/*библиотека vector для хранения данных на CPU*/
#include <vector>
/*библиотека random для генерации случайных чисел*/
#include <random>
/*библиотека algorithm для min/max*/
#include <algorithm>
/*библиотека cstdint для uint64_t*/
#include <cstdint>
/*библиотека string*/
#include <string>
/*библиотека functional для std::function*/
#include <functional>

/*макрос для проверки ошибок CUDA*/
#define CUDA_CHECK(call) do {                                 \
  cudaError_t err = call;                                     \
  if (err != cudaSuccess) {                                   \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                 \
            __FILE__, __LINE__, cudaGetErrorString(err));     \
    std::exit(1);                                             \
  }                                                           \
} while(0)

/*1) Генерация данных на CPU
  Заполняем вектор случайными числами в диапазоне [-1; 1]*/
static void fill_random(std::vector<float>& a, uint64_t seed=12345) {
  /*инициализируем генератор псевдослучайных чисел*/
  std::mt19937_64 rng(seed);
  /*распределение равномерное по [-1; 1]*/
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  /*заполняем массив*/
  for (auto &x : a) x = dist(rng);
}

/*2) Редукция суммы
  a) "только global"-каждый поток считает частичную сумму и делает atomicAdd в глобальную сумму
  b) global+shared-блочная редукция в shared и запись суммы блока в global*/

/*2a: редукция с использованием только глобальной памяти
  Каждый поток суммирует свой набор элементов (grid-stride),
  затем делает atomicAdd в глобальную переменную out_sum.*/
__global__ void reduce_global_atomic(const float* __restrict__ in, float* out_sum, int n) {
  /*глобальный индекс потока*/
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  /*локальная сумма в регистре*/
  float local = 0.0f;

  /*grid-stride loop чтобы покрыть весь массив даже при малой сетке*/
  for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
    local += in[i];
  }

  /*атомарное добавление в глобальную память*/
  atomicAdd(out_sum, local);
}

/*2b: редукция с использованием разделяемой памяти
  Каждый блок загружает по одному элементу на поток в shared,
  затем делает редукцию внутри блока и сохраняет сумму блока в block_sums.
  Финальную сумму (block_sums) считаем на CPU*/
__global__ void reduce_shared_block(const float* __restrict__ in, float* block_sums, int n) {
  /*динамическая разделяемая память, размер задаётся при запуске ядра*/
  extern __shared__ float sdata[];

  /*локальный индекс потока в блоке*/
  int tid = threadIdx.x;
  /*глобальный индекс элемента*/
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  /*загружаем элемент или 0 если вышли за границы*/
  float x = 0.0f;
  if (idx < n) x = in[idx];

  /*пишем в shared*/
  sdata[tid] = x;
  /*синхронизация, все потоки должны записать свои значения*/
  __syncthreads();

  /*редукция в shared, шаг уменьшается вдвое*/
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  /*поток 0 пишет сумму блока в глобальную память*/
  if (tid == 0) block_sums[blockIdx.x] = sdata[0];
}

/*3) Сортировка на GPU
Общий массив хранится в global memory
Внутри блока берём тайл (TILE элементов):
      1) каждый поток читает ITEMS_PER_THREAD элементов
      2) сортирует их пузырьком в локальном массиве (регистры/локальная память)
      3) пишет в shared
      4) затем выполняется слияние в shared (учебно: одним потоком)
Далее выполняются merge-проходы (парные слияния сегментов) с использованием shared*/

/*параметры тайла сортировки*/
constexpr int ITEMS_PER_THREAD = 8;        /*сколько элементов сортирует один поток локально*/
constexpr int THREADS          = 128;      /*потоков в блоке*/
constexpr int TILE             = THREADS * ITEMS_PER_THREAD; /*размер тайла*/

/*пузырьковая сортировка маленького массива в локальной памяти*/
__device__ void bubble_sort_local(float a[ITEMS_PER_THREAD]) {
  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    #pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD - 1 - i; j++) {
      if (a[j] > a[j+1]) {
        float t = a[j];
        a[j] = a[j+1];
        a[j+1] = t;
      }
    }
  }
}

/*сортировка одного тайла:
  1) load: global-local[ITEMS_PER_THREAD]
  2) bubble sort в local
  3) store: local-shared
  4) merge внутри shared до полного отсортированного тайла
  5) store: shared-global*/
__global__ void tile_sort_local_then_merge_shared(float* data, int n) {
  /*shared память для тайла и временного буфера*/
  __shared__ float sh[TILE];
  __shared__ float tmp[TILE];

  /*индекс потока*/
  int tid = threadIdx.x;
  /*база тайла в глобальном массиве*/
  int base = blockIdx.x * TILE;

  /*локальный массив потока*/
  float local[ITEMS_PER_THREAD];

  /*загрузка из global в local, за границами дополняем +inf*/
  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int g = base + tid * ITEMS_PER_THREAD + i;
    local[i] = (g < n) ? data[g] : INFINITY;
  }

  /*локальная сортировка пузырьком*/
  bubble_sort_local(local);

  /*запись в shared: у каждого потока своя полоса*/
  #pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    int s = tid * ITEMS_PER_THREAD + i;
    sh[s] = local[i];
  }
  __syncthreads();

  /*слияние отсортированных кусочков в shared, стартовый размер run = ITEMS_PER_THREAD, затем удваивается*/
  if (tid == 0) {
    int run = ITEMS_PER_THREAD;
    while (run < TILE) {
      for (int start = 0; start < TILE; start += 2 * run) {
        int mid = start + run;
        int end = start + 2 * run;

        int i = start, j = mid, k = start;

        /*классическое слияние двух отсортированных отрезков*/
        while (i < mid && j < end) {
          float a = sh[i];
          float b = sh[j];
          if (a <= b) { tmp[k++] = a; i++; }
          else        { tmp[k++] = b; j++; }
        }
        while (i < mid) tmp[k++] = sh[i++];
        while (j < end) tmp[k++] = sh[j++];

        /*копируем результат обратно в sh*/
        for (int t = start; t < end; t++) sh[t] = tmp[t];
      }
      run *= 2;
    }
  }
  __syncthreads();

  /*запись отсортированного тайла обратно в global*/
  for (int s = tid; s < TILE; s += blockDim.x) {
    int g = base + s;
    if (g < n) data[g] = sh[s];
  }
}

/*merge-pass: сливает пары отсортированных сегментов длины seg
вход: in (global)
выход: out (global)
shared используется как буфер для двух сегментов A и B*/
__global__ void merge_pass_shared(const float* in, float* out, int n, int seg) {
  /*shared буфер, сначала seg для A, затем seg для B*/
  extern __shared__ float buf[];
  float* A = buf;
  float* B = buf + seg;

  /*номер пары сегментов*/
  int pair = blockIdx.x;

  /*границы сегментов*/
  int start = pair * 2 * seg;
  int mid   = start + seg;
  int end   = min(start + 2 * seg, n);

  /*если старт за пределами массива то выходим*/
  if (start >= n) return;

  /*загрузка из global в shared, коалесцированно по threadIdx.x*/
  for (int i = threadIdx.x; i < seg; i += blockDim.x) {
    int gA = start + i;
    int gB = mid + i;
    A[i] = (gA < n) ? in[gA] : INFINITY;
    B[i] = (gB < n) ? in[gB] : INFINITY;
  }
  __syncthreads();

  /*слияние*/
  if (threadIdx.x == 0) {
    int i = 0, j = 0, k = start;

    /*реальные размеры сегментов у края массива*/
    int aN = min(seg, max(0, n - start));
    int bN = min(seg, max(0, n - mid));

    while (i < aN && j < bN) {
      float a = A[i], b = B[j];
      if (a <= b) out[k++] = a, i++;
      else        out[k++] = b, j++;
    }
    while (i < aN) out[k++] = A[i++];
    while (j < bN) out[k++] = B[j++];
  }
}

/*Таймер CUDA (cudaEvent)
Замеряем время одного запуска "launch()" в миллисекундах*/
static float time_kernel(std::function<void()> launch) {
  /*создаём события*/
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  /*старт*/
  CUDA_CHECK(cudaEventRecord(start));
  /*запуск ядра/набор ядер*/
  launch();
  /*стоп*/
  CUDA_CHECK(cudaEventRecord(stop));
  /*ждём завершения*/
  CUDA_CHECK(cudaEventSynchronize(stop));

  /*вычисляем время*/
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  /*удаляем события*/
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms;
}

/*Запуски, редукция и сортировка*/

/*структура строки результата*/
struct ResultRow {
  int n;
  float red_global_ms;
  float red_shared_ms;
  float sort_ms;
};

/*запуск редукции (global-only через atomicAdd)
выделяем d_sum, зануляем, запускаем ядро, читаем сумму на CPU*/
static float run_reduce_global(const float* d_in, int n) {
  /*указатель на сумму в глобальной памяти GPU*/
  float* d_sum = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
  CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

  /*конфигурация запуска*/
  int block = 256;
  int grid  = min(1024, (n + block - 1) / block);

  /*замер времени ядра*/
  float ms = time_kernel([&]{
    reduce_global_atomic<<<grid, block>>>(d_in, d_sum, n);
  });
  /*проверка запуска*/
  CUDA_CHECK(cudaGetLastError());
  /*ждём завершения*/
  CUDA_CHECK(cudaDeviceSynchronize());

  /*читаем результат*/
  float hsum = 0.0f;
  CUDA_CHECK(cudaMemcpy(&hsum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

  /*освобождаем память*/
  CUDA_CHECK(cudaFree(d_sum));
  return ms;
}

/*запуск редукции (global+shared)
ядро пишет суммы блоков в d_part, финальную сумму считаем на CPU*/
static float run_reduce_shared(const float* d_in, int n) {
  /*конфигурация запуска*/
  int block = 256;
  int grid  = (n + block - 1) / block;

  /*массив сумм блоков на GPU*/
  float* d_part = nullptr;
  CUDA_CHECK(cudaMalloc(&d_part, grid * sizeof(float)));

  /*замер времени ядра*/
  float ms = time_kernel([&]{
    reduce_shared_block<<<grid, block, block * sizeof(float)>>>(d_in, d_part, n);
  });
  /*проверка запуска*/
  CUDA_CHECK(cudaGetLastError());
  /*ждём завершения*/
  CUDA_CHECK(cudaDeviceSynchronize());

  /*копируем суммы блоков на CPU и суммируем*/
  std::vector<float> part(grid);
  CUDA_CHECK(cudaMemcpy(part.data(), d_part, grid * sizeof(float), cudaMemcpyDeviceToHost));

  volatile double sum = 0.0;
  for (float x : part) sum += x;

  /*освобождаем память*/
  CUDA_CHECK(cudaFree(d_part));
  return ms;
}

/*запуск пайплайна сортировки
  1.сортировка тайлов
  2.несколько merge-проходов*/
static float run_sort_pipeline(float* d_data, int n) {
  /*1) сортировка тайлов*/
  int blocks = (n + TILE - 1) / TILE;

  float ms1 = time_kernel([&]{
    tile_sort_local_then_merge_shared<<<blocks, THREADS>>>(d_data, n);
  });
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  /*2) буфер для merge-проходов*/
  float* d_tmp = nullptr;
  CUDA_CHECK(cudaMalloc(&d_tmp, n * sizeof(float)));

  float total_merge_ms = 0.0f;
  int seg = TILE;

  /*ограничение сегмента, чтобы shared помещался*/
  const int SEG_MAX = 4096;

  /*flip-флаг, чередуем вход/выход*/
  bool flip = false;

  while (seg < n && seg <= SEG_MAX) {
    /*число пар сегментов*/
    int pairs = (n + 2 * seg - 1) / (2 * seg);
    /*потоков в блоке*/
    int block = 256;
    /*shared на блок: 2*seg floats*/
    size_t shmem = 2ull * seg * sizeof(float);

    float pass_ms = time_kernel([&]{
      if (!flip) merge_pass_shared<<<pairs, block, shmem>>>(d_data, d_tmp, n, seg);
      else       merge_pass_shared<<<pairs, block, shmem>>>(d_tmp,  d_data, n, seg);
    });
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    total_merge_ms += pass_ms;
    flip = !flip;
    seg *= 2;
  }

  /*если финальный результат в d_tmp то копируем в d_data*/
  if (flip) {
    CUDA_CHECK(cudaMemcpy(d_data, d_tmp, n * sizeof(float), cudaMemcpyDeviceToDevice));
  }

  /*освобождаем временный буфер*/
  CUDA_CHECK(cudaFree(d_tmp));

  /*возвращаем суммарное время*/
  return ms1 + total_merge_ms;
}

/*функция main()
для размеров 10k, 100k, 1M:генерируем данные на CPU, копируем на GPU, меряем редукцию (global-only и shared),меряем сортировку
печатаем CSV, чтобы строить графики*/
int main() {
  /*размеры из задания*/
  std::vector<int> sizes = {10000, 100000, 1000000};

  /*таблица результатов*/
  std::vector<ResultRow> rows;
  rows.reserve(sizes.size());

  for (int n : sizes) {
    /*генерируем массив на CPU*/
    std::vector<float> h(n);
    fill_random(h, 12345);

    /*выделяем память на GPU под массив*/
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));

    /*копируем данные на GPU*/
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    /*редукция: global-only и global+shared*/
    float t_red_g = run_reduce_global(d, n);
    float t_red_s = run_reduce_shared(d, n);

    /*сортировка меняет массив на месте, поэтому копируем исходные данные снова*/
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    float t_sort = run_sort_pipeline(d, n);

    /*освобождаем память*/
    CUDA_CHECK(cudaFree(d));

    /*сохраняем строку результатов*/
    rows.push_back({n, t_red_g, t_red_s, t_sort});
  }

  /*вывод в формате CSV*/
  printf("n,reduce_global_ms,reduce_shared_ms,sort_ms\n");
  for (auto &r : rows) {
    printf("%d,%.6f,%.6f,%.6f\n", r.n, r.red_global_ms, r.red_shared_ms, r.sort_ms);
  }

  /*завершение программы*/
  return 0;
}
