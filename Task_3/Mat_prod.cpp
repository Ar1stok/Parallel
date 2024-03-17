#include <iostream>
#include <cmath>
#include <chrono>
#include <memory>
#include <thread>

#define arr_elem 20000
#define numThreads 16

std::shared_ptr<double[]> matrix(new double[arr_elem * arr_elem]);
std::shared_ptr<double[]> vector(new double[arr_elem]);
std::shared_ptr<double[]> answer(new double[arr_elem]);

void multiply (int start, int end)
{
    for (int i = start; i < end; i++) 
    {
        for (int j = 0; j < arr_elem; j++) 
        {
            answer[i] += matrix[i * arr_elem + j] * vector[j];
        }
    }
}

void init_matrix(int start, int end)
{
    for (int i = start; i < end; ++i)
    {
        matrix[i] = (i % (arr_elem + 1) == 0) ? 2.0 : 1.0;
    }
}

void init_vector(int start, int end)
{
    for (int i = start; i < end; ++i)
    {
        vector[i] = double(i) + 1.0;
    }
}

int main(int argc, char const* argv[])
{
    auto begin = std::chrono::steady_clock::now();
    
    std::jthread threads[numThreads];

    int size = arr_elem / numThreads;

    for (int i = 0; i < numThreads; ++i) // matrix init
    {
        int start_interval = i * size * arr_elem;
        int end_interval = (i == numThreads - 1) ? arr_elem * arr_elem : (i + 1) * size * arr_elem;
        threads[i] = std::jthread(init_matrix, start_interval, end_interval);
    }

    for (int i = 0; i < numThreads; ++i) // vector init 
    {
        int start_interval = i * size;
        int end_interval = (i == numThreads - 1) ? arr_elem : (i + 1) * size;
        threads[i] = std::jthread(init_vector, start_interval, end_interval);
    }

    for (int i = 0; i < numThreads; ++i) 
    {
        int start_interval = i * size;
        int end_interval = (i == numThreads - 1) ? arr_elem : (i + 1) * size;
        threads[i] = std::jthread(multiply, start_interval, end_interval);
    }

    for (int i = 0; i < numThreads; ++i) 
    {
        threads[i].join();
    }

    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

    std::cout << "The time: " << elapsed_ms.count() << " ms\n";

    return 0;
}