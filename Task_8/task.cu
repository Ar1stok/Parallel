#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <memory>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace po = boost::program_options;

#define at(arr, x, y) (arr[(x) * size + (y)])
#define size_sq size * size

constexpr int LEFT_UP = 10;
constexpr int LEFT_DOWN = 20;
constexpr int RIGHT_UP = 20;
constexpr int RIGHT_DOWN = 30;

template <class ctype>
class Data {
private:
    int len;
    ctype* d_arr;

public:
    std::vector<ctype> arr;

    Data(int length) : len(length), arr(len), d_arr(nullptr) {
        cudaError_t err = cudaMalloc((void**)&d_arr, len * sizeof(ctype));
        if (err != cudaSuccess) {
            std::cerr << "CUDA memory allocation failed: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    ~Data() {
        if (d_arr) {
            cudaFree(d_arr);
        }
    }

    void copyToDevice() {
        cudaError_t err = cudaMemcpy(d_arr, arr.data(), len * sizeof(ctype), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "CUDA memory copy to device failed: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void copyToHost() {
        cudaError_t err = cudaMemcpy(arr.data(), d_arr, len * sizeof(ctype), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "CUDA memory copy to host failed: " << cudaGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    ctype* getDevicePointer() {
        return d_arr;
    }
};


void initMatrix(std::vector<double>& mainArr, int size) {
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
}

void saveMatrix(const double* mainArr, int size, const std::string& filename) 
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

__global__ void iterate(double* matrix, double* lastMatrix, int size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j == 0 || i == 0 || i >= size - 1 || j >= size - 1) return;

    at(matrix, i, j) = 0.25 * (at(lastMatrix, i, j + 1) + at(lastMatrix, i, j - 1) +
                                at(lastMatrix, i - 1, j) + at(lastMatrix, i + 1, j));
}

template <unsigned int blockSize>
__global__ void compute_error(double* matrix, double* lastMatrix, double* errors, int size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= size || i >= size) return;

    // Allocate the temporary storage for a block of blockSize(32) threads of type double
    __shared__ typename cub::BlockReduce<double, blockSize>::TempStorage temp_storage;
    double local_max = 0.0;

    if (j > 0 && i > 0 && j < size - 1 && i < size - 1) {
        local_max = fabs(at(matrix, i, j) - at(lastMatrix, i, j));
    }

    // Calculate the largest value in the block using the reduction operation
    double block_max = cub::BlockReduce<double, blockSize>(temp_storage).Reduce(local_max, cub::Max());

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        errors[blockIdx.y * gridDim.x + blockIdx.x] = block_max;
    }
}


int main(int argc, char const *argv[]) {
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

    double error = 1.0;
    int iter = 0;

    Data<double> A(size_sq);
    Data<double> Anew(size_sq);

    initMatrix(A.arr, size);
    initMatrix(Anew.arr, size);

    auto start = std::chrono::high_resolution_clock::now();

    dim3 blockDim(32, 32);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x, (size + blockDim.y - 1) / blockDim.y);

    int totalBlocks = gridDim.x * gridDim.y;
    Data<double> errors(totalBlocks);

    A.copyToDevice();
    Anew.copyToDevice();
    errors.copyToDevice();

    double* A_link = A.getDevicePointer();
    double* Anew_link = Anew.getDevicePointer();
    double* errors_link = errors.getDevicePointer();

    std::unique_ptr<cudaStream_t, void(*)(cudaStream_t*)> 
    stream(new cudaStream_t, [](cudaStream_t* s) {
        cudaStreamDestroy(*s);
        delete s;
    });

    std::unique_ptr<cudaGraph_t, void(*)(cudaGraph_t*)> 
    graph(new cudaGraph_t, [](cudaGraph_t* g) {
        cudaGraphDestroy(*g);
        delete g;
    });

    std::unique_ptr<cudaGraphExec_t, void(*)(cudaGraphExec_t*)> 
    graphExec(new cudaGraphExec_t, [](cudaGraphExec_t* ge) {
        cudaGraphExecDestroy(*ge);
        delete ge;
    });

    cudaStreamCreate(stream.get());
    bool graphCreated = false;

    while (iter < iterations && error > eps) {
        if (!graphCreated) 
        {
            cudaStreamBeginCapture(*stream, cudaStreamCaptureModeGlobal);

            for (int i = 0; i < 999; i++) {
                iterate<<<gridDim, blockDim, 0, *stream>>>(A_link, Anew_link, size);
                std::swap(A_link, Anew_link);
            }

            iterate<<<gridDim, blockDim, 0, *stream>>>(A_link, Anew_link, size);
            compute_error<32><<<gridDim, blockDim, 0, *stream>>>(A_link, Anew_link, errors_link, size);

            cudaStreamEndCapture(*stream, graph.get());
            cudaGraphInstantiate(graphExec.get(), *graph, nullptr, nullptr, 0);

            graphCreated = true;
        } 
        else 
        {
            cudaGraphLaunch(*graphExec, *stream);
            cudaStreamSynchronize(*stream);

            errors.copyToHost();
            error = *std::max_element(errors.arr.begin(), errors.arr.end());
            iter += 1000;
        }
    }

    A.copyToHost();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Iterations: " << iter << "\n";
    std::cout << "Time: " << elapsed.count() << " s\n";
    std::cout << "Error: " << error << "\n";

    saveMatrix(A.arr.data(), size, "result_matrix.txt");

    return 0;
}



// __global__ - вызывается с хоста, запускает функцию на device
// каждый параллельный вызов ф-ции - это block
// набор таких блоков - grid
// block может быть разбит на потоки (threads)
// func <<<N, M>>> - запуск функции на gpu, где N - кол-во блоков, M - кол-во потоков 
// blockDim.x - кол-во потоков в блоке
// __shared__ используется для объявления переменной/массива в общей памяти
// dim3 blockDim(32, 32) - зависит от Warp Size (который у нас 32) группа потоков внутри потоковго блока, 
// которые физически выполняются одновременно