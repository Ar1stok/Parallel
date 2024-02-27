#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

#define arr_elem 500
#define numThreads 1
#define MAX_ITERATIONS 100000

int main(int argc, char const* argv[]) 
{

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
        vector_b[i] = double(arr_elem) + 1.0;
    }

    double eps = 0.001;

    for (int k = 0; k < MAX_ITERATIONS; ++k) 
    {
        double error = 0;
        double temp;

        for (int i = 0; i < arr_elem; i++) 
        {
            temp = x[i];
            x[i] = 0;
            for (int j = 0; j < arr_elem; j++) 
            {
                if (i != j)
                {
                    x[i] += matrix[i * arr_elem + j] * x[j];
                }
            }
            x[i] = (vector_b[i] - x[i]) / matrix[i * arr_elem + i];
            error = std::max(error, std::abs(temp - x[i]));
        }

        if (error < eps)
        {
            std::cout << k << "\n";
            break;
        }
    }

    return 0;
}