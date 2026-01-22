/*Assignment 5. Параллельные структуры данных на GPU (CUDA)
Часть 1. Реализация параллельного стека на CUDA
Задание: Реализовать структуру данных стек с использованием атомарных
операций для безопасного доступа к данным.
Задачи:
1. Инициализировать стек с фиксированной емкостью.
2. Написать ядро CUDA, использующее push и pop параллельно из
нескольких потоков.
3. Проверить корректность выполнения операций.
Часть 2. Реализация параллельной очереди на CUDA
Задание: Реализовать очередь с использованием атомарных операций для
безопасного добавления и удаления элементов.
Задачи:
1. Инициализировать очередь с заданной емкостью.
2. Написать ядро CUDA, использующее enqueue и dequeue параллельно.
3. Сравнить производительность очереди и стека.*/

/*подключаем CUDA Runtime API*/
#include <cuda_runtime.h>
/*библиотека для printf*/
#include <stdio.h>
/*vector для хранения результатов на CPU*/
#include <vector>
/*алгоритмы STL (min/max и т.п.)*/
#include <algorithm>
/*numeric (в данной работе почти не используется, но оставим как было)*/
#include <numeric>

/*макрос для проверки ошибок CUDA*/
#define CHECK_CUDA(call) do {                                 \
  cudaError_t err = call;                                     \
  if (err != cudaSuccess) {                                   \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
           cudaGetErrorString(err));                          \
    exit(1);                                                  \
  }                                                           \
} while(0)

/*Часть 1: Стек (LIFO)*/

/*Структура Stack хранится в global memory
data-буфер для элементов стека, top-индекс следующей свободной позиции,capacity-максимальная ёмкость стека*/
struct Stack {
  int *data;        /*буфер данных в global memory*/
  int top;          /*вершина: индекс следующей свободной ячейки*/
  int capacity;     /*максимальная ёмкость*/

  /*инициализация стека на устройстве*/
  __device__ void init(int *buffer, int size) {
    data = buffer;
    top = 0;              /*пустой стек - size = 0*/
    capacity = size;
  }

  /*push: добавляем элемент в стек, используем atomicAdd чтобы несколько потоков не писали в одну ячейку*/
  __device__ bool push(int value) {
    int pos = atomicAdd(&top, 1);  /*получаем позицию куда писать*/
    if (pos < capacity) {
      data[pos] = value;
      return true;
    }
    /*если переполнение то откатываем top назад*/
    atomicSub(&top, 1);
    return false;
  }

  /*pop: достаём элемент из стека
    atomicSub уменьшает top и возвращает старое значение.
    pos = atomicSub 1 даёт индекс последнего элемента*/
  __device__ bool pop(int *value) {
    int pos = atomicSub(&top, 1) - 1;
    if (pos >= 0) {
      *value = data[pos];
      return true;
    }
    /*если стек пустой то откат*/
    atomicAdd(&top, 1);
    return false;
  }
};

/*ядро инициализации стека,запускаем 1 блок и 1 поток, чтобы корректно инициализировать поля*/
__global__ void stack_init_kernel(Stack *s, int *buffer, int capacity) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    s->init(buffer, capacity);
  }
}

/*ядро тестирования стека:
1.PUSH каждый поток кладёт tid в стек
2.POP каждый поток пытается достать значение из стека
  pop_out[tid]-результат pop для потока tid
  push_ok/pop_ok-счётчики успешных операций*/
__global__ void stack_push_pop_kernel(Stack *s, int *pop_out, int n_ops,
                                      int *push_ok, int *pop_ok) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  /*Фаза PUSH*/
  if (tid < n_ops) {
    bool ok = s->push(tid);
    if (ok) atomicAdd(push_ok, 1);
  }

  /*синхронизация потоков внутри блока*/
  __syncthreads();

  /*Фаза POP*/
  if (tid < n_ops) {
    int v;
    bool ok = s->pop(&v);
    if (ok) {
      pop_out[tid] = v;
      atomicAdd(pop_ok, 1);
    } else {
      /*если не удалось достать элемент*/
      pop_out[tid] = -1;
    }
  }
}

/*Часть 2: Очередь (FIFO)*/

/*реализация FIFO очереди*/
struct Queue {
  int *data;        /*буфер данных*/
  int head;         /*индекс чтения*/
  int tail;         /*индекс записи*/
  int capacity;     /*ёмкость очереди*/

  /*инициализация очереди*/
  __device__ void init(int *buffer, int size) {
    data = buffer;
    head = 0;
    tail = 0;
    capacity = size;
  }

  /*enqueue: добавляем элемент (FIFO)
    atomicAdd по tail выдаёт уникальную позицию записи*/
  __device__ bool enqueue(int value) {
    int pos = atomicAdd(&tail, 1);
    if (pos < capacity) {
      data[pos] = value;
      return true;
    }
    /*overflow - откат*/
    atomicSub(&tail, 1);
    return false;
  }

  /*dequeue: забираем элемент
    atomicAdd по head выдаёт уникальную позицию чтения, для корректности проверяем pos < tail*/
  __device__ bool dequeue(int *value) {
    int pos = atomicAdd(&head, 1);
    if (pos < tail) {
      *value = data[pos];
      return true;
    }
    /*underflow - откат*/
    atomicSub(&head, 1);
    return false;
  }
};

/*ядро инициализации очереди*/
__global__ void queue_init_kernel(Queue *q, int *buffer, int capacity) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    q->init(buffer, capacity);
  }
}

/*ядро enqueue: каждый поток пытается добавить tid*/
__global__ void queue_enqueue_kernel(Queue *q, int n_ops, int *enq_ok) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n_ops) {
    bool ok = q->enqueue(tid);
    if (ok) atomicAdd(enq_ok, 1);
  }
}

/*ядро dequeue: каждый поток пытается извлечь элемент*/
__global__ void queue_dequeue_kernel(Queue *q, int *deq_out, int n_ops, int *deq_ok) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n_ops) {
    int v;
    bool ok = q->dequeue(&v);
    if (ok) {
      deq_out[tid] = v;
      atomicAdd(deq_ok, 1);
    } else {
      deq_out[tid] = -1;
    }
  }
}

/*измерение времени*/

/*функция для вычисления времени между двумя CUDA событиями*/
float elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

/*функция main()*/
int main() {
  /*параметры эксперимента*/
  const int threads = 256;             /*потоков в блоке*/
  const int blocks  = 120;             /*число блоков*/
  const int N       = threads * blocks;/*число операций*/
  const int CAP     = N;               /*ёмкость (можно уменьшить для overflow)*/

  printf("N ops=%d, capacity=%d, grid=(%d,%d)\n", N, CAP, blocks, threads);

  /*Память под STACK*/
  Stack *d_stack;            /*указатель на структуру Stack на GPU*/
  int *d_stack_buf;          /*буфер данных стека*/
  int *d_stack_pop_out;      /*массив результатов pop*/
  int *d_push_ok, *d_pop_ok; /*счётчики успешных операций*/

  CHECK_CUDA(cudaMalloc(&d_stack, sizeof(Stack)));
  CHECK_CUDA(cudaMalloc(&d_stack_buf, CAP * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_stack_pop_out, N * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_push_ok, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_pop_ok, sizeof(int)));
  CHECK_CUDA(cudaMemset(d_push_ok, 0, sizeof(int)));
  CHECK_CUDA(cudaMemset(d_pop_ok, 0, sizeof(int)));

  /*инициализация стека*/
  stack_init_kernel<<<1,1>>>(d_stack, d_stack_buf, CAP);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  /*Память под QUEUE*/
  Queue *d_queue;            /*структура Queue на GPU*/
  int *d_queue_buf;          /*буфер данных очереди*/
  int *d_queue_out;          /*массив результатов dequeue*/
  int *d_enq_ok, *d_deq_ok;  /*счётчики успехов*/

  CHECK_CUDA(cudaMalloc(&d_queue, sizeof(Queue)));
  CHECK_CUDA(cudaMalloc(&d_queue_buf, CAP * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_queue_out, N * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_enq_ok, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_deq_ok, sizeof(int)));
  CHECK_CUDA(cudaMemset(d_enq_ok, 0, sizeof(int)));
  CHECK_CUDA(cudaMemset(d_deq_ok, 0, sizeof(int)));

  /*инициализация очереди*/
  queue_init_kernel<<<1,1>>>(d_queue, d_queue_buf, CAP);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  /*CUDA events для тайминга*/
  cudaEvent_t s0,s1,q0,q1;
  CHECK_CUDA(cudaEventCreate(&s0));
  CHECK_CUDA(cudaEventCreate(&s1));
  CHECK_CUDA(cudaEventCreate(&q0));
  CHECK_CUDA(cudaEventCreate(&q1));

  /*Запуск теста STACK*/
  CHECK_CUDA(cudaEventRecord(s0));
  stack_push_pop_kernel<<<blocks, threads>>>(d_stack, d_stack_pop_out, N, d_push_ok, d_pop_ok);
  CHECK_CUDA(cudaEventRecord(s1));
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventSynchronize(s1));

  float stack_ms = elapsed_ms(s0,s1);

  /*копируем счётчики на CPU*/
  int h_push_ok=0, h_pop_ok=0;
  CHECK_CUDA(cudaMemcpy(&h_push_ok, d_push_ok, sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(&h_pop_ok,  d_pop_ok,  sizeof(int), cudaMemcpyDeviceToHost));

  /*копируем массив pop-результатов на CPU*/
  std::vector<int> h_stack_out(N);
  CHECK_CUDA(cudaMemcpy(h_stack_out.data(), d_stack_pop_out, N*sizeof(int), cudaMemcpyDeviceToHost));

  /*подсчитываем количество валидных значений*/
  int valid_stack = 0;
  for (int v : h_stack_out) if (v >= 0) valid_stack++;

  printf("\n[STACK]\n");
  printf("push_ok=%d, pop_ok=%d, host_valid_pop=%d\n", h_push_ok, h_pop_ok, valid_stack);
  printf("time = %.3f ms\n", stack_ms);

  /*проверка диапазона значений: должны быть [-1] или [0..N-1]*/
  bool stack_range_ok = true;
  for (int v : h_stack_out) {
    if (v != -1 && (v < 0 || v >= N)) { stack_range_ok = false; break; }
  }
  printf("range_check: %s\n", stack_range_ok ? "OK" : "FAIL");

  /*Запуск теста QUEUE (enqueue затем dequeue)*/
  CHECK_CUDA(cudaEventRecord(q0));
  queue_enqueue_kernel<<<blocks, threads>>>(d_queue, N, d_enq_ok);
  queue_dequeue_kernel<<<blocks, threads>>>(d_queue, d_queue_out, N, d_deq_ok);
  CHECK_CUDA(cudaEventRecord(q1));
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventSynchronize(q1));

  float queue_ms = elapsed_ms(q0,q1);

  /*копируем счётчики*/
  int h_enq_ok=0, h_deq_ok=0;
  CHECK_CUDA(cudaMemcpy(&h_enq_ok, d_enq_ok, sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(&h_deq_ok, d_deq_ok, sizeof(int), cudaMemcpyDeviceToHost));

  /*копируем результаты dequeue*/
  std::vector<int> h_queue_out(N);
  CHECK_CUDA(cudaMemcpy(h_queue_out.data(), d_queue_out, N*sizeof(int), cudaMemcpyDeviceToHost));

  /*сколько валидных значений*/
  int valid_queue = 0;
  for (int v : h_queue_out) if (v >= 0) valid_queue++;

  printf("\n[QUEUE]\n");
  printf("enq_ok=%d, deq_ok=%d, host_valid_deq=%d\n", h_enq_ok, h_deq_ok, valid_queue);
  printf("time = %.3f ms\n", queue_ms);

  /*проверка диапазона значений*/
  bool queue_range_ok = true;
  for (int v : h_queue_out) {
    if (v != -1 && (v < 0 || v >= N)) { queue_range_ok = false; break; }
  }
  printf("range_check: %s\n", queue_range_ok ? "OK" : "FAIL");

  /*Сравнение производительности*/
  printf("\n[PERFORMANCE]\n");
  printf("stack kernel time: %.3f ms\n", stack_ms);
  printf("queue kernels time: %.3f ms\n", queue_ms);
  printf("ratio queue/stack: %.3f\n", (queue_ms > 0 ? queue_ms/stack_ms : 0));

  /*Освобождение памяти и ресурсов*/

  CHECK_CUDA(cudaFree(d_stack));
  CHECK_CUDA(cudaFree(d_stack_buf));
  CHECK_CUDA(cudaFree(d_stack_pop_out));
  CHECK_CUDA(cudaFree(d_push_ok));
  CHECK_CUDA(cudaFree(d_pop_ok));

  CHECK_CUDA(cudaFree(d_queue));
  CHECK_CUDA(cudaFree(d_queue_buf));
  CHECK_CUDA(cudaFree(d_queue_out));
  CHECK_CUDA(cudaFree(d_enq_ok));
  CHECK_CUDA(cudaFree(d_deq_ok));

  CHECK_CUDA(cudaEventDestroy(s0));
  CHECK_CUDA(cudaEventDestroy(s1));
  CHECK_CUDA(cudaEventDestroy(q0));
  CHECK_CUDA(cudaEventDestroy(q1));

  return 0;
}
