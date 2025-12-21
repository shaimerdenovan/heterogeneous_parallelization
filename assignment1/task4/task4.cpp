/*Задание 4
Создайте массив из 5 000 000 чисел и реализуйте вычисление среднего значения
элементов массива последовательным способом и с использованием OpenMP с редукцией.
Сравните время выполнения обеих реализаций.*/

/*подключаем библиотеку для ввода и вывода данных*/
#include <iostream>                 
/*библиотека для генерации случайных чисел*/
#include <random>   
/*библиотека для измерения времени выполнения*/
#include <chrono>
/*библиотека для параллельных вычислений OpenMP*/
#include <omp.h>

/*используем стандартное пространство имён чтоб не писать std:: каждый раз*/
using namespace std;

/*главная функция*/
int main() {

    /*размер массива устанавливаем 5000000 элементов*/
    const int SIZE = 5000000;

    /*динамически выделяем память под массив*/
    int* array = new int[SIZE];

    /*создаем источник случайных чисел*/
    random_device rd;
    /*генератор случайных чисел*/
    mt19937 gen(rd());
    /*задаем диапазон от 1 до 100*/
    uniform_int_distribution<int> dist(1, 100);

    /*заполняем массив случайными числами*/
    for (int i = 0; i < SIZE; i++) {
        array[i] = dist(gen);
    }

    /*переменная для суммы в последовательном варианте*/
    long long seq_sum = 0;

    /*запоминаем время начала последовательного вычисления*/
    auto startSeq = chrono::high_resolution_clock::now();

    /*последовательно суммируем элементы массива*/
    for (int i = 0; i < SIZE; i++) {
        seq_sum += array[i];
    }

    /*запоминаем время окончания последовательного вычисления*/
    auto endSeq = chrono::high_resolution_clock::now();

    /*вычисляем среднее значение для последовательного варианта*/
    double seq_avg = (double)seq_sum / SIZE;

    /*вычисляем время выполнения последовательного варианта*/
    chrono::duration<double> seq_time = endSeq - startSeq;

    /*переменная для суммы в параллельном варианте*/
    long long par_sum = 0;

    /*запоминаем время начала параллельного вычисления*/
    auto par_start = chrono::high_resolution_clock::now();

    /*параллельно суммируем элементы массива с использованием OpenMP*/
    #pragma omp parallel for reduction(+:par_sum)
    for (int i = 0; i < SIZE; i++) {
        par_sum += array[i];
    }

    /*запоминаем время окончания параллельного вычисления*/
    auto par_end = chrono::high_resolution_clock::now();

    /*вычисляем среднее значение для параллельного варианта*/
    double par_avg = (double)par_sum / SIZE;

    /*вычисляем время выполнения параллельного варианта*/
    chrono::duration<double> par_time = par_end - par_start;

    /*выводим результаты последовательного варианта*/
    cout << "The sequential average value is: " << seq_avg << endl;
    cout << "The sequential time is: " << seq_time.count() << endl;
    /*выводим результаты параллельного варианта*/
    cout << "The parallel average value is: " << par_avg << endl;
    cout << "The parallel time is: " << par_time.count() << endl;

    /*освобождаем динамически выделенную память*/
    delete[] array;

    /*обнуляем указатель*/
    array = nullptr;

    /*завершение программы*/
    return 0;
}