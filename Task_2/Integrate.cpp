#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

#define numThreads 40

double func(double x) 
{
    return exp(-x * x);
}

double midpointRectangleIntegration(double a, double b, int n) 
{
    double h = (b - a) / n;
    double sum = 0.0;

    #pragma omp parallel num_threads(numThreads)
    {
        double temp;

        #pragma omp for schedule(dynamic, 10000)
        for (int i = 0; i < n; i++) 
        {
            double x_midpoint = a + h/2 + i*h;
            temp += func(x_midpoint);
        }

        #pragma omp atomic
        sum += temp;
    }

    return h * sum;
}

int main(int argc, char const* argv[]) {
    double a = -4.0; // Нижний предел интегрирования
    double b = 4.0; // Верхний предел интегрирования
    int nsteps = 40000000; // Количество прямоугольников

    auto begin = std::chrono::steady_clock::now();
    double result = midpointRectangleIntegration(a, b, nsteps);
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

    std::cout << "The time: " << elapsed_ms.count() << " ms\n";
    std::cout << "Result:" << result << std::endl;
    
    return 0;
}