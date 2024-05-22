#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>
#include <math.h>
#include <cmath>

#ifdef OPENACC__
#include <openacc.h>
#endif
#ifdef NVPROF_
#include </opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include/nvtx3/nvToolsExt.h>
#endif
#include <omp.h>

#define at(arr, x, y) (arr[(x) * size + (y)])
#define size_sq size * size

constexpr int LEFT_UP = 10;
constexpr int LEFT_DOWN = 20;
constexpr int RIGHT_UP = 20;
constexpr int RIGHT_DOWN = 30;
constexpr int ITERS_BETWEEN_UPDATE = 70;

void initArrays(double *mainArr, double *subArr, int &size)
{
    std::memset(mainArr, 0, sizeof(double) * size_sq);

    at(mainArr, 0, 0) = LEFT_UP;
    at(mainArr, 0, size - 1) = RIGHT_UP;
    at(mainArr, size - 1, 0) = LEFT_DOWN;
    at(mainArr, size - 1, size - 1) = RIGHT_DOWN;

    for (int i = 1; i < size - 1; i++)
    {
        at(mainArr, 0, i) = (at(mainArr, 0, size - 1) - at(mainArr, 0, 0)) / (size - 1) * i + at(mainArr, 0, 0);
        at(mainArr, i, 0) = (at(mainArr, size - 1, 0) - at(mainArr, 0, 0)) / (size - 1) * i + at(mainArr, 0, 0);

        at(mainArr, size - 1, i) = (at(mainArr, size - 1, size - 1) - at(mainArr, size - 1, 0)) / (size - 1) * i + at(mainArr, size - 1, 0);
        at(mainArr, i, size - 1) = (at(mainArr, size - 1, size - 1) - at(mainArr, 0, size - 1)) / (size - 1) * i + at(mainArr, 0, size - 1);
    }
    std::memcpy(subArr, mainArr, sizeof(double) * size_sq);
}

void saveMatrix(double *mainArr, int size, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) 
    {
        std::cerr << "Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    for (int i = 0; i < size; ++i) 
    {
        for (int j = 0; j < size; ++j) 
        {
            outputFile << at(mainArr, i, j) << ' ';
        }
        outputFile << std::endl;
    }

    outputFile.close();
}


int main(int argc, char *argv[])
{

    bool showResult = false;
    double eps = 1E-6;
    int iterations = 1E6;
    int size = 10;

    for(int arg = 0; arg < argc; arg++)
    {
        std::stringstream stream;
        if(strcmp(argv[arg], "-eps") == 0)
        {
            stream << argv[arg+1];
            stream >> eps;
        }
        else if(strcmp(argv[arg], "-s") == 0)
        {
            stream << argv[arg+1];
            stream >> size;
        }
        else if(strcmp(argv[arg], "-show") == 0)
        {
            stream << argv[arg+1];
            stream >> showResult;
        }
    }
    std::cout << "Current settings:" << std::endl;
    std::cout << "\tEPS: " << eps << std::endl;
    std::cout << "\tMax iteration: " << iterations << std::endl;
    std::cout << "\tSize: " << size << 'x' << size << std::endl;

    double start = omp_get_wtime();

    double *F = new double[size_sq];
    double *Fnew = new double[size_sq];

    initArrays(F, Fnew, size);

    double error = 0;
    int iteration = 0;

    int itersBetweenUpdate = 0;

#pragma acc enter data copyin(Fnew[:size_sq], F[:size_sq], error)

#ifdef NVPROF_
    nvtxRangePush("MainCycle");
#endif
    do
    {
        #pragma acc parallel present(error) async
        {
            error = 0;
        }

        #pragma acc parallel loop collapse(2) present(Fnew[:size_sq], F[:size_sq], error) vector_length(128) async
        for (int x = 1; x < size - 1; x++)
        {
            for (int y = 1; y < size - 1; y++)
            {
                at(Fnew, x, y) = 0.25 * (at(F, x + 1, y) + at(F, x - 1, y) + at(F, x, y - 1) + at(F, x, y + 1));
            }
        }
        
        std::swap(F, Fnew);
        
#ifdef OPENACC__
        acc_attach((void **)F);
        acc_attach((void **)Fnew);
#endif
        if (itersBetweenUpdate >= ITERS_BETWEEN_UPDATE && iteration < iterations)
        {
            #pragma acc parallel loop collapse(2) present(Fnew[:size_sq], F[:size_sq], error) reduction(max:error) vector_length(128) async
            for (int x = 1; x < size - 1; x++)
            {
                for (int y = 1; y < size - 1; y++)
                {
                    error = fmax(error, fabs(at(Fnew, x, y) - at(F, x, y)));
                }
            }
            #pragma acc update self(error) wait
            itersBetweenUpdate = -1;
        }
        else
        {
            error = 1;
        }
        iteration++;
        itersBetweenUpdate++;
    } while (iteration < iterations && error > eps);
#ifdef NVPROF_
    nvtxRangePop();
#endif

    #pragma acc parallel loop collapse(2) present(Fnew[:size_sq], F[:size_sq], error) reduction(max:error) vector_length(128) async
    for (int x = 1; x < size - 1; x++)
    {
        for (int y = 1; y < size - 1; y++)
        {
            error = fmax(error, fabs(at(Fnew, x, y) - at(F, x, y)));
        }
    }
    #pragma acc update self(error)
    #pragma acc exit data delete (Fnew[:size_sq]) copyout(F[:size_sq], error)

    double end = omp_get_wtime();
    std::cout << "Time: " << end - start << " s" << std::endl;
    std::cout << "Iterations: " << iteration << std::endl;
    std::cout << "Error: " << error << std::endl;
    if (showResult) saveMatrix(F, size, "matrix.txt");

    delete[] F;
    delete[] Fnew;

    return 0;
}