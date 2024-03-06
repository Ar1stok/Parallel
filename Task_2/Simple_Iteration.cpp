#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <memory>

#define arr_elem 1000
#define numThreads 16

int main(int argc, char const* argv[]) 
{
    std::shared_ptr<double[]> matrix(new double[arr_elem * arr_elem]);
    std::shared_ptr<double[]> vector_x(new double[arr_elem]);
    std::shared_ptr<double[]> vector_b(new double[arr_elem]);
    std::shared_ptr<double[]> temp_values(new double[arr_elem]);

    double eps = 0.00001;
    double t = 0.00001;
    double term = 0.0;
    double norm_v_b = sqrt((double(arr_elem) + 1.0) * (double(arr_elem) + 1.0) * double(arr_elem));

    auto begin = std::chrono::steady_clock::now();
    #pragma omp parallel num_threads(numThreads)
    {
        #pragma omp for
        for (int i = 0; i < arr_elem * arr_elem; i++)
        {
            matrix[i] = (i % (arr_elem + 1) == 0) ? 2.0 : 1.0;
        }

        #pragma omp for
        for (int i = 0; i < arr_elem; i++)
        {
            vector_b[i] = double(arr_elem) + 1.0;
        }

        #pragma omp for
        for (int i = 0; i < arr_elem; i++)
        {
            vector_x[i] = 0.0;
        }

        #pragma omp for
        for (int i = 0; i < arr_elem; i++)
        {
            temp_values[i] = 0.0;
        }
        
        while (true) {

            #pragma omp for schedule(dynamic, 10)
            for (int i = 0; i < arr_elem; ++i) 
            {
                double sum = 0;
                for (int j = 0; j < arr_elem; ++j) 
                {
                    sum += matrix[i * arr_elem + j] * temp_values[j];
                }
                vector_x[i] = temp_values[i] - t * (sum - vector_b[i]);
            }
            
            // Критерий остановки
            term = 0.0;

            #pragma omp for
            for (int i = 0; i < arr_elem; ++i) 
            {
                double sum = 0;
                for (int j = 0; j < arr_elem; ++j) 
                {
                    sum += matrix[i * arr_elem + j] * vector_x[j];
                }
                term += (sum - vector_b[i]) * (sum - vector_b[i]);
            }

            if (std::abs(sqrt(term) / norm_v_b) < eps)
            {
                break;
            }
            
            for (int i = 0; i < arr_elem; i++)
            {
                temp_values[i] = vector_x[i];
            }
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

    std::cout << "Vector: " << vector_x[0];

    std::cout << "\n The time: " << elapsed_ms.count() << " ms\n";

    return 0;
}