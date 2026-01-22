/*Assignment 5 (Дополнительные задания)
Дополнительные задания
1. Реализовать очередь с поддержкой нескольких производителей и
потребителей (MPMC).
2. Оптимизировать использование памяти, включая работу с разделяемой
памятью.
3. Сравнить производительность реализованных структур данных с
последовательными версиями.*/

/*подключаем CUDA Runtime API*/
#include <cuda_runtime.h>
/*библиотека для printf*/
#include <stdio.h>
/*vector для хранения результатов на CPU*/
#include <vector>
/*chrono для измерения времени на CPU*/
#include <chrono>

/*макрос проверки ошибок CUDA*/
#define CHECK_CUDA(call) do {                                 \
  cudaError_t err = call;                                     \
  if (err != cudaSuccess) {                                   \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
           cudaGetErrorString(err));                          \
    exit(1);                                                  \
  }                                                           \
} while(0)

/*функция для вычисления времени между CUDA событиями*/
static inline float elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

/*Часть 1: STACK (atomic) + STACK batched push (shared memory)*/

/*структура стека в global memory*/
struct Stack {
  int *data;        /*буфер данных*/
  int top;          /*размер стека (индекс следующей свободной позиции)*/
  int capacity;     /*ёмкость*/

  /*инициализация*/
  __device__ void init(int *buffer, int size) {
    data = buffer;
    top = 0;
    capacity = size;
  }

  /*push: атомарно увеличиваем top и записываем элемент*/
  __device__ bool push(int value) {
    int pos = atomicAdd(&top, 1);
    if (pos < capacity) {
      data[pos] = value;
      return true;
    }
    /*overflow*/
    atomicSub(&top, 1);
    return false;
  }

  /*pop: атомарно уменьшаем top и читаем элемент*/
  __device__ bool pop(int *value) {
    int pos = atomicSub(&top, 1) - 1;
    if (pos >= 0) {
      *value = data[pos];
      return true;
    }
    /*underflow*/
    atomicAdd(&top, 1);
    return false;
  }
};

/*ядро инициализации стека (1 поток)*/
__global__ void stack_init_kernel(Stack *s, int *buffer, int capacity) {
  if (blockIdx.x == 0 && threadIdx.x == 0) s->init(buffer, capacity);
}

/*корректные фазы push/pop выполняем разными ядрами*/

/*ядро push: каждый поток кладёт tid*/
__global__ void stack_push_kernel(Stack *s, int n_ops, int *push_ok) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n_ops) {
    if (s->push(tid)) atomicAdd(push_ok, 1);
  }
}

/*ядро pop: каждый поток пытается достать значение*/
__global__ void stack_pop_kernel(Stack *s, int *pop_out, int n_ops, int *pop_ok) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n_ops) {
    int v;
    bool ok = s->pop(&v);
    pop_out[tid] = ok ? v : -1;
    if (ok) atomicAdd(pop_ok, 1);
  }
}

/*batched push: уменьшаем число атомиков
  1) каждый поток кладёт данные в shared
  2) поток 0 считает, сколько потоков активны в блоке (block_count)
  3) поток 0 делает atomicAdd(&top, block_count) и получает base
  4) каждый поток пишет в global по позиции base + local_index*/
__global__ void stack_batched_push_kernel(Stack *s, int n_ops, int *push_ok) {
  extern __shared__ int sh[]; /*shared для значений потока*/
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int active = 0;
  if (tid < n_ops) {
    sh[threadIdx.x] = tid;
    active = 1;
  }
  __syncthreads();

  /*считаем количество активных потоков в блоке*/
  __shared__ int block_count;
  if (threadIdx.x == 0) {
    int cnt = 0;
    int base_tid = blockIdx.x * blockDim.x;
    for (int i = 0; i < blockDim.x; i++) cnt += ((base_tid + i) < n_ops);
    block_count = cnt;
  }
  __syncthreads();

  /*одна атомарная операция на блок: резервируем block_count позиций*/
  __shared__ int base;
  if (threadIdx.x == 0) base = atomicAdd(&s->top, block_count);
  __syncthreads();

  /*каждый активный поток пишет свой элемент*/
  if (active) {
    int local = threadIdx.x;
    int pos = base + local;
    if (local < block_count && pos < s->capacity) {
      s->data[pos] = sh[local];
      atomicAdd(push_ok, 1);
    }
  }
}

/*Часть 2: QUEUE (simple) + QUEUE batched enqueue (shared memory)*/

/*очередь FIFO в global memory*/
struct Queue {
  int *data;        /*буфер данных*/
  int head;         /*индекс чтения*/
  int tail;         /*индекс записи*/
  int capacity;     /*ёмкость*/

  /*инициализация*/
  __device__ void init(int *buffer, int size) {
    data = buffer;
    head = 0;
    tail = 0;
    capacity = size;
  }

  /*enqueue: атомарно берём позицию tail и записываем*/
  __device__ bool enqueue(int value) {
    int pos = atomicAdd(&tail, 1);
    if (pos < capacity) {
      data[pos] = value;
      return true;
    }
    /*overflow*/
    atomicSub(&tail, 1);
    return false;
  }

  /*dequeue: атомарно берём позицию head и читаем, условие pos < tail означает что элемент уже был добавлен*/
  __device__ bool dequeue(int *value) {
    int pos = atomicAdd(&head, 1);
    if (pos < tail) {
      *value = data[pos];
      return true;
    }
    /*underflow*/
    atomicSub(&head, 1);
    return false;
  }
};

/*ядро инициализации очереди*/
__global__ void queue_init_kernel(Queue *q, int *buffer, int capacity) {
  if (blockIdx.x == 0 && threadIdx.x == 0) q->init(buffer, capacity);
}

/*ядро enqueue*/
__global__ void queue_enqueue_kernel(Queue *q, int n_ops, int *enq_ok) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n_ops) {
    if (q->enqueue(tid)) atomicAdd(enq_ok, 1);
  }
}

/*ядро dequeue*/
__global__ void queue_dequeue_kernel(Queue *q, int *out, int n_ops, int *deq_ok) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n_ops) {
    int v;
    bool ok = q->dequeue(&v);
    out[tid] = ok ? v : -1;
    if (ok) atomicAdd(deq_ok, 1);
  }
}

/*batched enqueue: 1 atomicAdd на блок,механизм аналогичен stack_batched_push_kernel*/
__global__ void queue_batched_enqueue_kernel(Queue *q, int n_ops, int *enq_ok) {
  extern __shared__ int sh[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int active = 0;
  if (tid < n_ops) {
    sh[threadIdx.x] = tid;
    active = 1;
  }
  __syncthreads();

  __shared__ int block_count;
  if (threadIdx.x == 0) {
    int cnt = 0;
    int base_tid = blockIdx.x * blockDim.x;
    for (int i = 0; i < blockDim.x; i++) cnt += ((base_tid + i) < n_ops);
    block_count = cnt;
  }
  __syncthreads();

  __shared__ int base;
  if (threadIdx.x == 0) base = atomicAdd(&q->tail, block_count);
  __syncthreads();

  if (active) {
    int local = threadIdx.x;
    int pos = base + local;
    if (local < block_count && pos < q->capacity) {
      q->data[pos] = sh[local];
      atomicAdd(enq_ok, 1);
    }
  }
}

/*MPMC bounded queue (Vyukov-style)+корректный тест*/

/*MPMC очередь: несколько производителей и потребителей, используется кольцевой буфер (ring) и массив seq (sequence numbers)*/
struct MPMCQueue {
  int *data;               /*ring buffer*/
  unsigned int *seq;       /*sequence number per slot*/
  unsigned int head;       /*индекс чтения*/
  unsigned int tail;       /*индекс записи*/
  int capacity;            /*ёмкость (размер кольца)*/

  /*инициализация указателей*/
  __device__ void init_ptrs(int *buf, unsigned int *seqbuf, int cap) {
    data = buf;
    seq  = seqbuf;
    head = 0;
    tail = 0;
    capacity = cap;
  }

  /*enqueue: несколько производителей*/
  __device__ bool enqueue(int value) {
    unsigned int cap = (unsigned int)capacity;
    while (true) {
      unsigned int pos = tail;
      unsigned int idx = pos % cap;

      unsigned int s = seq[idx];
      int dif = (int)s - (int)pos;

      if (dif == 0) {
        /*слот свободен, пытаемся забронировать tail*/
        if (atomicCAS(&tail, pos, pos + 1) == pos) {
          data[idx] = value;
          __threadfence();      /*гарантируем запись data перед записью seq*/
          seq[idx] = pos + 1;   /*слот теперь готов для dequeue*/
          return true;
        }
      } else if (dif < 0) {
        /*очередь заполнена*/
        return false;
      }
      /*иначе кто-то обогнал - повторяем*/
    }
  }

  /*dequeue: несколько потребителей*/
  __device__ bool dequeue(int *value) {
    unsigned int cap = (unsigned int)capacity;
    while (true) {
      unsigned int pos = head;
      unsigned int idx = pos % cap;

      unsigned int s = seq[idx];
      int dif = (int)s - (int)(pos + 1);

      if (dif == 0) {
        /*слот готов, пытаемся забронировать head*/
        if (atomicCAS(&head, pos, pos + 1) == pos) {
          int v = data[idx];
          __threadfence();      /*гарантируем чтение data до изменения seq*/
          seq[idx] = pos + cap; /*слот снова свободен для enqueue следующего круга*/
          *value = v;
          return true;
        }
      } else if (dif < 0) {
        /*очередь пуста*/
        return false;
      }
    }
  }
};

/*ядро инициализации указателей MPMCQueue*/
__global__ void mpmc_init_ptrs_kernel(MPMCQueue *q, int *buf, unsigned int *seqbuf, int cap) {
  if (blockIdx.x == 0 && threadIdx.x == 0) q->init_ptrs(buf, seqbuf, cap);
}

/*инициализация массива seq: seq[i] = i*/
__global__ void mpmc_init_seq_kernel(unsigned int *seq, int cap) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < cap) seq[tid] = (unsigned int)tid;
}

/*kernel производителей:каждый поток делает iters попыток enqueue*/
__global__ void mpmc_producers_kernel(MPMCQueue *q, int iters, int *enq_ok) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < iters; i++) {
    int val = tid * iters + i;
    if (q->enqueue(val)) atomicAdd(enq_ok, 1);
  }
}

/*kernel потребителей:нужно извлечь ровно total_to_take элементов в массив out
deq_ok используется как глобальный счётчик уже извлечённых элементов*/
__global__ void mpmc_consumers_kernel(MPMCQueue *q, int *out, int out_cap,
                                     int total_to_take, int *deq_ok) {
  while (true) {
    int cur = atomicAdd(deq_ok, 0);
    if (cur >= total_to_take) break;

    int v;
    if (q->dequeue(&v)) {
      int pos = atomicAdd(deq_ok, 1);
      if (pos < out_cap) out[pos] = v;
    }
  }
}

/*Доп.3: CPU sequential baselines*/

/*последовательный стек на CPU: push N элементов и pop N элементов*/
static double cpu_stack_time_ms(int N) {
  std::vector<int> st;
  st.reserve(N);

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++) st.push_back(i);

  volatile int sink = 0;
  for (int i = 0; i < N; i++) {
    sink ^= st.back();
    st.pop_back();
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  (void)sink;

  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

/*последовательная очередь на CPU: массив+head/tail*/
static double cpu_queue_time_ms(int N) {
  std::vector<int> q(N);
  int head = 0, tail = 0;

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < N; i++) q[tail++] = i;

  volatile int sink = 0;
  for (int i = 0; i < N; i++) sink ^= q[head++];
  auto t1 = std::chrono::high_resolution_clock::now();
  (void)sink;

  return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

/*функция Main*/
int main() {
  /*параметры запуска на GPU*/
  const int threads = 256;
  const int blocks  = 120;
  const int N       = threads * blocks; /*общее число потоков/операций*/
  const int CAP     = N;                /*ёмкость буферов*/

  printf("N=%d, CAP=%d, grid=(%d,%d)\n", N, CAP, blocks, threads);

  /*STACK allocations*/
  Stack *d_stack;          /*структура Stack на GPU*/
  int *d_stack_buf;        /*буфер данных*/
  int *d_stack_pop;        /*выход pop*/
  int *d_push_ok, *d_pop_ok; /*счётчики успехов*/

  CHECK_CUDA(cudaMalloc(&d_stack, sizeof(Stack)));
  CHECK_CUDA(cudaMalloc(&d_stack_buf, CAP*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_stack_pop, N*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_push_ok, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_pop_ok, sizeof(int)));

  /*инициализация стека*/
  stack_init_kernel<<<1,1>>>(d_stack, d_stack_buf, CAP);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  /*QUEUE allocations*/
  Queue *d_q;              /*структура Queue на GPU*/
  int *d_q_buf;            /*буфер*/
  int *d_q_out;            /*выход dequeue*/
  int *d_enq_ok, *d_deq_ok;/*счётчики успехов*/

  CHECK_CUDA(cudaMalloc(&d_q, sizeof(Queue)));
  CHECK_CUDA(cudaMalloc(&d_q_buf, CAP*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_q_out, N*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_enq_ok, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_deq_ok, sizeof(int)));

  /*инициализация очереди*/
  queue_init_kernel<<<1,1>>>(d_q, d_q_buf, CAP);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  /*MPMC allocations*/
  MPMCQueue *d_mq;          /*структура MPMCQueue на GPU*/
  int *d_mq_buf;            /*ring buffer*/
  unsigned int *d_mq_seq;   /*sequence array*/
  int *d_mpmc_out;          /*результаты dequeue*/
  int *d_m_enq_ok;          /*счётчик успешных enqueue*/
  int *d_m_deq_ok;          /*счётчик успешных dequeue*/

  CHECK_CUDA(cudaMalloc(&d_mq, sizeof(MPMCQueue)));
  CHECK_CUDA(cudaMalloc(&d_mq_buf, CAP*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_mq_seq, CAP*sizeof(unsigned int)));
  CHECK_CUDA(cudaMalloc(&d_mpmc_out, N*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_m_enq_ok, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_m_deq_ok, sizeof(int)));

  /*CUDA events*/
  cudaEvent_t e0,e1;
  CHECK_CUDA(cudaEventCreate(&e0));
  CHECK_CUDA(cudaEventCreate(&e1));

  /*STACK atomic (2 kernels: push then pop)*/
  CHECK_CUDA(cudaMemset(d_push_ok, 0, sizeof(int)));
  CHECK_CUDA(cudaMemset(d_pop_ok,  0, sizeof(int)));

  stack_init_kernel<<<1,1>>>(d_stack, d_stack_buf, CAP);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(e0));

  stack_push_kernel<<<blocks,threads>>>(d_stack, N, d_push_ok);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  stack_pop_kernel<<<blocks,threads>>>(d_stack, d_stack_pop, N, d_pop_ok);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(e1));
  CHECK_CUDA(cudaEventSynchronize(e1));

  float stack_atomic_ms = elapsed_ms(e0,e1);

  int h_push_ok=0,h_pop_ok=0;
  CHECK_CUDA(cudaMemcpy(&h_push_ok, d_push_ok, sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(&h_pop_ok,  d_pop_ok,  sizeof(int), cudaMemcpyDeviceToHost));

  printf("\n[STACK atomic]\n");
  printf("push_ok=%d pop_ok=%d time=%.3f ms\n", h_push_ok, h_pop_ok, stack_atomic_ms);

  /*STACK batched push (shared memory) (только push)*/
  CHECK_CUDA(cudaMemset(d_push_ok, 0, sizeof(int)));

  stack_init_kernel<<<1,1>>>(d_stack, d_stack_buf, CAP);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(e0));

  stack_batched_push_kernel<<<blocks,threads,threads*sizeof(int)>>>(d_stack, N, d_push_ok);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(e1));
  CHECK_CUDA(cudaEventSynchronize(e1));

  float stack_batched_ms = elapsed_ms(e0,e1);

  int h_push_ok2=0;
  CHECK_CUDA(cudaMemcpy(&h_push_ok2, d_push_ok, sizeof(int), cudaMemcpyDeviceToHost));

  printf("\n[STACK batched push (shared memory)]\n");
  printf("push_ok=%d time=%.3f ms\n", h_push_ok2, stack_batched_ms);

  /*QUEUE simple atomic (enqueue then dequeue)*/
  CHECK_CUDA(cudaMemset(d_enq_ok, 0, sizeof(int)));
  CHECK_CUDA(cudaMemset(d_deq_ok, 0, sizeof(int)));

  queue_init_kernel<<<1,1>>>(d_q, d_q_buf, CAP);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(e0));

  queue_enqueue_kernel<<<blocks,threads>>>(d_q, N, d_enq_ok);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  queue_dequeue_kernel<<<blocks,threads>>>(d_q, d_q_out, N, d_deq_ok);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(e1));
  CHECK_CUDA(cudaEventSynchronize(e1));

  float queue_atomic_ms = elapsed_ms(e0,e1);

  int h_enq_ok=0,h_deq_ok=0;
  CHECK_CUDA(cudaMemcpy(&h_enq_ok, d_enq_ok, sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(&h_deq_ok, d_deq_ok, sizeof(int), cudaMemcpyDeviceToHost));

  printf("\n[QUEUE simple atomic]\n");
  printf("enq_ok=%d deq_ok=%d time=%.3f ms\n", h_enq_ok, h_deq_ok, queue_atomic_ms);

  /*QUEUE batched enqueue (shared memory) + dequeue*/
  CHECK_CUDA(cudaMemset(d_enq_ok, 0, sizeof(int)));
  CHECK_CUDA(cudaMemset(d_deq_ok, 0, sizeof(int)));

  queue_init_kernel<<<1,1>>>(d_q, d_q_buf, CAP);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(e0));

  queue_batched_enqueue_kernel<<<blocks,threads,threads*sizeof(int)>>>(d_q, N, d_enq_ok);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  queue_dequeue_kernel<<<blocks,threads>>>(d_q, d_q_out, N, d_deq_ok);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(e1));
  CHECK_CUDA(cudaEventSynchronize(e1));

  float queue_batched_ms = elapsed_ms(e0,e1);

  int h_enq_ok2=0,h_deq_ok2=0;
  CHECK_CUDA(cudaMemcpy(&h_enq_ok2, d_enq_ok, sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(&h_deq_ok2, d_deq_ok, sizeof(int), cudaMemcpyDeviceToHost));

  printf("\n[QUEUE batched enqueue (shared memory)]\n");
  printf("enq_ok=%d deq_ok=%d time=%.3f ms\n", h_enq_ok2, h_deq_ok2, queue_batched_ms);

  /*MPMC queue: producers kernel-consumers kernel*/
  CHECK_CUDA(cudaMemset(d_m_enq_ok, 0, sizeof(int)));
  CHECK_CUDA(cudaMemset(d_m_deq_ok, 0, sizeof(int)));

  /*инициализация указателей на буферы*/
  mpmc_init_ptrs_kernel<<<1,1>>>(d_mq, d_mq_buf, d_mq_seq, CAP);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  /*инициализация seq: seq[i] = i*/
  int init_blocks = (CAP + threads - 1) / threads;
  mpmc_init_seq_kernel<<<init_blocks, threads>>>(d_mq_seq, CAP);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  /*сколько операций делает один producer thread*/
  const int iters = 8; /*можно увеличить (например 64/256) для более стабильных замеров*/

  CHECK_CUDA(cudaEventRecord(e0));

  /*1) producers*/
  mpmc_producers_kernel<<<blocks,threads>>>(d_mq, iters, d_m_enq_ok);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  /*узнаём сколько реально добавили*/
  int h_m_enq = 0;
  CHECK_CUDA(cudaMemcpy(&h_m_enq, d_m_enq_ok, sizeof(int), cudaMemcpyDeviceToHost));

  /*2) consumers: забираем ровно h_m_enq элементов*/
  mpmc_consumers_kernel<<<blocks,threads>>>(d_mq, d_mpmc_out, N, h_m_enq, d_m_deq_ok);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(e1));
  CHECK_CUDA(cudaEventSynchronize(e1));

  float mpmc_ms = elapsed_ms(e0,e1);

  int h_m_deq = 0;
  CHECK_CUDA(cudaMemcpy(&h_m_deq, d_m_deq_ok, sizeof(int), cudaMemcpyDeviceToHost));

  printf("\n[MPMC queue]\n");
  printf("iters=%d total_threads=%d\n", iters, N);
  printf("enq_ok=%d deq_ok=%d time=%.3f ms\n", h_m_enq, h_m_deq, mpmc_ms);

  /*CPU sequential baselines*/
  double cpu_stack_ms = cpu_stack_time_ms(N);
  double cpu_queue_ms = cpu_queue_time_ms(N);

  printf("\n[CPU sequential]\n");
  printf("stack push+pop: %.3f ms\n", cpu_stack_ms);
  printf("queue enq+deq : %.3f ms\n", cpu_queue_ms);

  /*Итоговое сравнение*/
  printf("\n[SUMMARY]\n");
  printf("GPU stack atomic              : %.3f ms\n", stack_atomic_ms);
  printf("GPU stack batched (SMEM)      : %.3f ms\n", stack_batched_ms);
  printf("GPU queue atomic (2 kernels)  : %.3f ms\n", queue_atomic_ms);
  printf("GPU queue batched enqueue     : %.3f ms\n", queue_batched_ms);
  printf("GPU MPMC queue (2 kernels)    : %.3f ms\n", mpmc_ms);
  printf("CPU stack sequential          : %.3f ms\n", cpu_stack_ms);
  printf("CPU queue sequential          : %.3f ms\n", cpu_queue_ms);

  /*Освобождение памяти и ресурсов*/

  CHECK_CUDA(cudaFree(d_stack));
  CHECK_CUDA(cudaFree(d_stack_buf));
  CHECK_CUDA(cudaFree(d_stack_pop));
  CHECK_CUDA(cudaFree(d_push_ok));
  CHECK_CUDA(cudaFree(d_pop_ok));

  CHECK_CUDA(cudaFree(d_q));
  CHECK_CUDA(cudaFree(d_q_buf));
  CHECK_CUDA(cudaFree(d_q_out));
  CHECK_CUDA(cudaFree(d_enq_ok));
  CHECK_CUDA(cudaFree(d_deq_ok));

  CHECK_CUDA(cudaFree(d_mq));
  CHECK_CUDA(cudaFree(d_mq_buf));
  CHECK_CUDA(cudaFree(d_mq_seq));
  CHECK_CUDA(cudaFree(d_mpmc_out));
  CHECK_CUDA(cudaFree(d_m_enq_ok));
  CHECK_CUDA(cudaFree(d_m_deq_ok));

  CHECK_CUDA(cudaEventDestroy(e0));
  CHECK_CUDA(cudaEventDestroy(e1));

  return 0;
}
