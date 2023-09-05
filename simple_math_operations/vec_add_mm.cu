#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>

// CUDA kernel for two vectors addition
__global__ void vectorAdd(const int *__restrict a, const int *__restrict b,
                          int *__restrict c, int N) {
    // Calculate global thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (tid < N)
        c[tid] = a[tid] + b[tid];
}

// Check vector containing results with addition of two vectors
void verifyResult(const int* a, const int* b, const int* c, int s) {
    for (int i = 0; i < s; i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

// Generate random numbers to fill vector
void initWithRandom(int *vec, int length){
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(1, 100);

    // fill vector
    for(int i = 0; i < length; i++){
        vec[i] = distribution(generator);
    }
}

void showSamples(int* vec){
    for(int i = 0; i < 50; i++){
        std::cout << vec[i] << "\t";
        if(i % 5 == 4){
            std::cout << std::endl;
        }
    }
}

int main() {
    // Array size of 2^24 (16777216 elements)
    constexpr int N = 1 << 24;
    constexpr size_t b_size = sizeof(int) * N;

    // GPU id
    int gpuID;
    int allDevices;
    cudaGetDeviceCount(&allDevices);
    cudaError_t err = cudaGetDevice(&gpuID);
    if (err != cudaSuccess) {
        std::cout << "Failed to get current device: " << cudaGetErrorString(err) << std::endl;
    }

    // Pointers for holding the data
    int *dataA, *dataB, *result;


    // Allocate memory on the device (GPU)
    err = cudaMallocManaged(&dataA, b_size);
    if (err != cudaSuccess) {
        std::cout << "Error dataA allocation: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMallocManaged(&dataB, b_size);
    if (err != cudaSuccess) {
        std::cout << "Error dataB allocation: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaMallocManaged(&result, b_size);
    if (err != cudaSuccess) {
        std::cout << "Error result allocation: " << cudaGetErrorString(err) << std::endl;
    }

    // Initialize random numbers in each array
    initWithRandom(dataA, N);
    initWithRandom(dataB, N);

    // Threads per block (1024)
    int NUM_THREADS = 1 << 10;

    // Blocks number
    int NUM_BLOCKS = static_cast<int>((N + NUM_THREADS - 1) / NUM_THREADS);

    // inform GPU it will need those (works only for Linux)
//    err = cudaMemPrefetchAsync(dataA, b_size, gpuID);
//    if (err != cudaSuccess) {
//        std::cout << "Error when prefetch: " << cudaGetErrorString(err) << std::endl;
//    }
//    err = cudaMemPrefetchAsync(dataB, b_size, gpuID);
//    if (err != cudaSuccess) {
//        std::cout << "Error when prefetch: " << cudaGetErrorString(err) << std::endl;
//    }

    // Launch the kernel on the GPU
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(dataA, dataB, result, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error when run vectorAdd: " << cudaGetErrorString(err) << std::endl;
    }

    // now result will be needed
    cudaMemPrefetchAsync(result, b_size, cudaCpuDeviceId);

    // Wait for all operations done
    cudaDeviceSynchronize();

    // Check result for errors
    verifyResult(dataA, dataB, result, N);

    // Show samples
    std::cout << "SOME SAMPLES FROM RESULT VECTOR:" << std::endl;
    showSamples(result);

    // Free memory on device
    cudaFree(dataA);
    cudaFree(dataB);
    cudaFree(result);

    std::cout << "\nCOMPLETED SUCCESSFULLY\n";

    return 0;
}