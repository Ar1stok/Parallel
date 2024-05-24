#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <sstream>
#include <memory>
#include <math.h>
#include <cmath>
#include <boost/program_options.hpp>

#ifdef OPENACC__
#include <openacc.h>
#endif
#ifdef NVPROF_
#include </opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include/nvtx3/nvToolsExt.h>
#endif
#include <omp.h>
namespace po = boost::program_options;

#define at(arr, x, y) (arr[(x) * size + (y)])
#define size_sq size * size

constexpr int LEFT_UP = 10;
constexpr int LEFT_DOWN = 20;
constexpr int RIGHT_UP = 20;
constexpr int RIGHT_DOWN = 30;
constexpr int ITERS_BETWEEN_UPDATE = 70;

void initArrays(double* mainArr, double* subArr, int &size, bool& initMean)
{
    std::memset(mainArr, 0, sizeof(double) * size_sq);

    // Заполнение матрицы средними значениями
    for (int i = 0; i < size_sq && initMean; i++)
    {
        mainArr[i] = (LEFT_UP + LEFT_DOWN + RIGHT_UP + RIGHT_DOWN) / 4;
    }

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

void saveMatrix(double* mainArr, int size, const std::string& filename) 
{
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
            outputFile << std::setw(4) << std::fixed << std::setprecision(4) << at(mainArr, i, j) << ' ';
        }
        outputFile << std::endl;
    }
    outputFile.close();
}


int main(int argc, char *argv[])
{
    po::options_description desc("options");
    desc.add_options()
        ("eps", po::value<double>()->default_value(1e-6),"Accuracy")
        ("size", po::value<int>()->default_value(10),"Matrix size")
        ("iterations", po::value<int>()->default_value(1000000),"Max count of iteration")
        ("show", po::value<bool>()->default_value(false),"Show ResMatrix")
        ("init", po::value<bool>()->default_value(false),"Use mean value during init")
        ("help", "Show all all command")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    double eps = vm["eps"].as<double>();
    int size = vm["size"].as<int>();
    int iterations = vm["iterations"].as<int>();
    bool showResult = vm["show"].as<bool>();
    bool initMean = vm["init"].as<bool>();

    std::cout << "Current settings:" << std::endl;
    std::cout << "\tEPS: " << eps << std::endl;
    std::cout << "\tMax iteration: " << iterations << std::endl;
    std::cout << "\tSize: " << size << 'x' << size << std::endl;
    std::cout << "\tMean Value: " << initMean << std::endl;

    double start = omp_get_wtime();

    std::shared_ptr<double[]> ArrF(new double[size_sq]);
    std::shared_ptr<double[]> ArrFnew(new double[size_sq]);

    initArrays(ArrF.get(), ArrFnew.get(), size, initMean);

    double* F = ArrF.get();
    double* Fnew = ArrFnew.get();

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

        // Распараллеливаем вложенные циклы parallel loop collapse(2)
        // (present - сообщают компилятору, что данные на устройстве)
        #pragma acc parallel loop collapse(2) present(Fnew[:size_sq], F[:size_sq], error) async
        for (int x = 1; x < size - 1; x++)
        {
            for (int y = 1; y < size - 1; y++)
            {
                at(Fnew, x, y) = 0.25 * (at(F, x + 1, y) + at(F, x - 1, y) + at(F, x, y - 1) + at(F, x, y + 1));
            }
        }
        
        double *swap = F;
        F = Fnew;
        Fnew = swap;

#ifdef OPENACC__
        acc_attach((void **)F);
        acc_attach((void **)Fnew);
#endif
        if (itersBetweenUpdate >= ITERS_BETWEEN_UPDATE && iteration < iterations)
        {
            // Вычисление ошибки (Используем редукцию)
            #pragma acc parallel loop collapse(2) present(Fnew[:size_sq], F[:size_sq], error) reduction(max:error) async
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

    // Последнее вычисление ошибки (Т.к был добавлен itersBetweenUpdate)
    #pragma acc parallel loop collapse(2) present(Fnew[:size_sq], F[:size_sq], error) reduction(max:error) async
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
    if (showResult) saveMatrix(ArrF.get(), size, "matrix.txt");

    return 0;
}