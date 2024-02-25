#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

#define arr_elem 3
#define numThreads 1

int main(int argc, char const* argv[]) {

    double *matrix, *vector_b;
    matrix = new double[arr_elem * arr_elem];
    vector_b = new double[arr_elem];
    double x[arr_elem] = {0.0};
    
    for (int i = 0; i < arr_elem * arr_elem; i++)
    {
        matrix[i] = (i % (arr_elem + 1) == 0) ? 2.0 : 1.0;
    }

    for (int i = 0; i < arr_elem; i++)
    {
        vector_b[i] = double(i) + 1.0;
    }

    double t = 0.1;
    double eps = 0.001;
    double x_new[arr_elem] = {0.0}; // Начальное приближение

    // Выполняем итерации
    for (int k = 0; k < 20; k++)
    {
        for (int i = 0; i < arr_elem; i++) 
        {
            for (int j = 0; j < arr_elem; j++) 
            {
                x_new[i] += x[i] - t * (matrix[i * arr_elem + j] * x[i] - vector_b[j]);
            }
        }

        for (int i = 0; i < arr_elem; i++)
        {
            std::cout << x_new[i] << " ";
        }
        std::cout << "\n";
        for (int i = 0; i < arr_elem; i++)
        {
            std::cout << x[i] << " ";
        }
        std::cout << "\n";
        // Проверяем условие сходимости
        double error = 0.0;
        for (int i = 0; i < arr_elem; i++) 
        {
            error += std::abs(x_new[i] - vector_b[i]) / vector_b[i];
        }

        if (error < eps) 
        {
            break;
        }

        for (int i = 0; i < arr_elem; i++) 
        {
            x[i] = x_new[i];
        }
    }

    std::cout << "Solution: ";
    for (int i = 0; i < arr_elem; i++) {
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}