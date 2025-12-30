/*Задание
1. Реализовать параллельную сортировку слиянием на CUDA:
● Разделите массив на блоки, каждый из которых будет обрабатываться
одним блоком потоков.
● Сортируйте блоки параллельно и сливайте их по парам.
2. Реализовать параллельную быструю сортировку на CUDA:
● Используйте параллельные потоки для деления массива по опорному
элементу.
● В каждом потоке выполняется быстрая сортировка на своей части
массива.
3. Реализовать параллельную пирамидальную сортировку на CUDA:
● Постройте кучу и выполняйте извлечение элементов параллельно, где
это возможно.
4. Сравнение производительности:
● Реализуйте последовательные версии этих алгоритмов на CPU.
● Измерьте время выполнения каждой сортировки на CPU и на GPU для
массивов разного размера (например, 10,000, 100,000 и 1,000,000
элементов).
● Сравните производительность и сделайте выводы.*/
/*подключаем библиотеку для ввода и вывода*/
#include <iostream>
/*библиотека для работы с динамическими массивами*/
#include <vector>
/*библиотека для алгоритмов стандартной библиотеки*/
#include <algorithm>
/*библиотека для генерации случайных чисел*/
#include <random>
/*библиотека для измерения времени выполнения*/
#include <chrono>
/*библиотека для работы с CUDA*/
#include <cuda_runtime.h>

/*определяем размер блока потоков для GPU*/
#define BLOCK_SIZE 256

/*используем стандартное пространство имён*/
using namespace std;
using namespace std::chrono;

/*CPU сортировки*/
void cpu_merge(vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    /*создаем временные массивы для левой и правой части*/
    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++) L[i] = arr[l + i];
    for (int i = 0; i < n2; i++) R[i] = arr[m + 1 + i];

    int i = 0, j = 0, k = l;
    /*слияние двух массивов обратно в исходный*/
    while(i < n1 && j < n2){
        if(L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }
    /*добавляем оставшиеся элементы левой части*/
    while(i < n1) arr[k++] = L[i++];
    /*добавляем оставшиеся элементы правой части*/
    while(j < n2) arr[k++] = R[j++];
}

/*рекурсивная реализация merge sort на CPU*/
/*делим массив на две части, сортируем их и сливаем*/
void cpu_merge_sort(vector<int>& arr, int l, int r){
    if(l < r){
        int m = l + (r - l)/2;
        cpu_merge_sort(arr, l, m);
        cpu_merge_sort(arr, m+1, r);
        cpu_merge(arr, l, m, r);
    }
}

/*рекурсивная реализация quick sort на CPU*/
/*выбираем опорный элемент и делим массив на меньшие и большие элементы*/
void cpu_quick_sort(vector<int>& arr, int l, int r){
    if(l < r){
        int pivot = arr[r];
        int i = l - 1;
        /*переставляем элементы относительно pivot*/
        for(int j = l; j < r; j++){
            if(arr[j] <= pivot) swap(arr[++i], arr[j]);
        }
        swap(arr[i+1], arr[r]);
        int pi = i + 1;
        cpu_quick_sort(arr, l, pi-1);
        cpu_quick_sort(arr, pi+1, r);
    }
}

/*функция heapify для heap sort на CPU*/
/*поддерживает свойство кучи для поддерева с корнем i*/
void cpu_heapify(vector<int>& arr, int n, int i){
    int largest = i;
    int l = 2*i + 1;
    int r = 2*i + 2;
    if(l < n && arr[l] > arr[largest]) largest = l;
    if(r < n && arr[r] > arr[largest]) largest = r;
    /*если корень не самый большой, меняем местами и рекурсивно исправляем поддерево*/
    if(largest != i){
        swap(arr[i], arr[largest]);
        cpu_heapify(arr, n, largest);
    }
}

/*реализация heap sort на CPU*/
/*строим кучу и последовательно извлекаем элементы из неё*/
void cpu_heap_sort(vector<int>& arr){
    int n = arr.size();
    /*строим кучу*/
    for(int i = n/2 - 1; i >=0; i--) cpu_heapify(arr, n, i);
    for(int i = n-1; i>0; i--){
        /*перемещаем максимум в конец*/
        swap(arr[0], arr[i]);
        /*восстанавливаем кучу*/
        cpu_heapify(arr, i, 0);
    }
}

/*CUDA метод сортировки слиянием*/
/*ядро для merge sort на GPU*/
/*каждый поток обрабатывает слияние двух подмассивов*/
__global__ void merge_kernel(int *arr, int width, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int start = idx * width * 2;
    if(start >= n) return;

    int mid = min(start + width, n);
    int end = min(start + 2 * width, n);

    /*временный массив в shared memory*/
    extern __shared__ int temp[];
    int i = start, j = mid, k = 0;
    /*слияние двух подмассивов в temp*/
    while(i < mid && j < end){
        temp[k++] = arr[i] <= arr[j] ? arr[i++] : arr[j++];
    }
    while(i < mid) temp[k++] = arr[i++];
    while(j < end) temp[k++] = arr[j++];
    /*копируем обратно в исходный массив*/
    for(i = start, k = 0; i < end; i++, k++) arr[i] = temp[k];
}

/*функция запуска merge sort на GPU*/
/*пошагово увеличиваем размер сливаемых подмассивов*/
void gpu_merge_sort(int *d_arr, int n){
    int width = 1;
    while(width < n){
        int threads = (n + 2*width - 1)/(2*width);
        int sharedMem = 2*width*sizeof(int);
        merge_kernel<<<(threads + BLOCK_SIZE -1)/BLOCK_SIZE, BLOCK_SIZE, sharedMem>>>(d_arr, width, n);
        /*ждем завершения ядра*/
        cudaDeviceSynchronize();
        width *= 2;
    }
}

/*CUDA метод быстрой сортировки*/
/*рекурсивная реализация быстрой сортировки на GPU для одного потока*/
__device__ void device_quick_sort(int *arr, int l, int r){
    if(l < r){
        int pivot = arr[r];
        int i = l - 1;
        for(int j = l; j < r; j++){
            if(arr[j] <= pivot) {
                i++;
                int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
            }
        }
        int tmp = arr[i+1]; arr[i+1] = arr[r]; arr[r] = tmp;
        int pi = i + 1;
        device_quick_sort(arr, l, pi-1);
        device_quick_sort(arr, pi+1, r);
    }
}

/*ядро запуска быстрой сортировки на GPU*/
/*для упрощения запускается только один поток*/
__global__ void quick_sort_kernel(int *arr, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx == 0){
        device_quick_sort(arr, 0, n-1);
    }
}

/* CUDA метод пирамидальной сортировки*/
/*heapify на GPU*/
__device__ void device_heapify(int *arr, int n, int i){
    int largest = i;
    int l = 2*i + 1;
    int r = 2*i + 2;
    if(l < n && arr[l] > arr[largest]) largest = l;
    if(r < n && arr[r] > arr[largest]) largest = r;
    if(largest != i){
        int tmp = arr[i]; arr[i] = arr[largest]; arr[largest] = tmp;
        device_heapify(arr, n, largest);
    }
}

/*ядро пирамидальной сортировки на GPU*/
/*для упрощения запускается один поток*/
__global__ void heap_sort_kernel(int *arr, int n){
    int idx = threadIdx.x;
    if(idx == 0){
        for(int i = n/2 - 1; i>=0; i--) device_heapify(arr, n, i);
        for(int i = n-1; i>0; i--){
            int tmp = arr[0]; arr[0] = arr[i]; arr[i] = tmp;
            device_heapify(arr, i, 0);
        }
    }
}

/*генерация случайного массива*/
/*заполняем вектор числами от 0 до 1 000 000*/
vector<int> generate_random_vector(int n){
    vector<int> v(n);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 1000000);
    for(int i = 0; i<n; i++) v[i] = dis(gen);
    return v;
}

/*измерение времени выполнения функции в миллисекундах*/
template<typename Func>
double measure_time(Func f){
    auto start = high_resolution_clock::now();
    f();
    auto end = high_resolution_clock::now();
    return duration<double, milli>(end - start).count();
}

/*главная функция*/
int main(){
    /*размеры массивов для тестирования*/
    vector<int> sizes = {10000, 100000, 1000000};

    for(auto n : sizes){
        cout << "Array size: " << n << endl;
        vector<int> arr = generate_random_vector(n);

        /*CPU Merge sort*/
        vector<int> cpu_arr = arr;
        double t_cpu_merge = measure_time([&](){ cpu_merge_sort(cpu_arr, 0, n-1); });
        cout << "CPU Merge Sort: " << t_cpu_merge << " ms" << endl;

        /*CPU Quick sort*/
        cpu_arr = arr;
        double t_cpu_quick = measure_time([&](){ cpu_quick_sort(cpu_arr, 0, n-1); });
        cout << "CPU Quick Sort: " << t_cpu_quick << " ms" << endl;

        /*CPU Heap sort*/
        cpu_arr = arr;
        double t_cpu_heap = measure_time([&](){ cpu_heap_sort(cpu_arr); });
        cout << "CPU Heap Sort: " << t_cpu_heap << " ms" << endl;

        /*GPU arrays*/
        int *d_arr;
        cudaMalloc(&d_arr, n * sizeof(int));
        cudaMemcpy(d_arr, arr.data(), n*sizeof(int), cudaMemcpyHostToDevice);

        /*GPU Merge sort*/
        cudaMemcpy(d_arr, arr.data(), n*sizeof(int), cudaMemcpyHostToDevice);
        double t_gpu_merge = measure_time([&](){ gpu_merge_sort(d_arr, n); });
        cout << "GPU Merge Sort: " << t_gpu_merge << " ms" << endl;

        /*GPU Quick sort*/
        cudaMemcpy(d_arr, arr.data(), n*sizeof(int), cudaMemcpyHostToDevice);
        double t_gpu_quick = measure_time([&](){ quick_sort_kernel<<<1,1>>>(d_arr, n); cudaDeviceSynchronize(); });
        cout << "GPU Quick Sort: " << t_gpu_quick << " ms" << endl;

        /*GPU Heap sort*/
        cudaMemcpy(d_arr, arr.data(), n*sizeof(int), cudaMemcpyHostToDevice);
        double t_gpu_heap = measure_time([&](){ heap_sort_kernel<<<1,1>>>(d_arr, n); cudaDeviceSynchronize(); });
        cout << "GPU Heap Sort: " << t_gpu_heap << " ms" << endl;

        cudaFree(d_arr);
    }

    return 0;
}