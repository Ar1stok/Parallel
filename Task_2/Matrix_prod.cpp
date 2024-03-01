#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <memory>

#define arr_elem 20000
#define numThreads 16

int main(int argc, char const* argv[])
{
    std::shared_ptr<double[]> matrix(new double[arr_elem * arr_elem]);
    std::shared_ptr<double[]> vector(new double[arr_elem]);
    std::shared_ptr<double[]> answer(new double[arr_elem]);

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
            vector[i] = double(i) + 1.0;
        }

        #pragma omp for collapse(2) nowait
        for (int i = 0; i < arr_elem; i++) 
        {
            for (int j = 0; j < arr_elem; j++) 
            {
                answer[i] += matrix[i * arr_elem + j] * vector[j];
            }
        }
    }
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

    std::cout << "The time: " << elapsed_ms.count() << " ms\n";

    return 0;
}