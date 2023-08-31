#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
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
void verifyResult(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c) {
    const size_t s = a.size();
    for (int i = 0; i < s; i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

// Generate random numbers to fill vector
void generateRandom(std::vector<int> &vec){
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(1, 100);

    // fill vector
    std::generate(vec.begin(), vec.end(), [&]() { return distribution(generator); });
}

void showSamples(std::vector<int> &vec){
    for(int i = 0; i < 50; i++){
        std::cout << vec[i] << "\t";
        if(i % 5 == 4){
            std::cout << std::endl;
        }
    }
}

int main() {
    // Array size of 2^24 (16777216 elements)
    constexpr unsigned long long N = 1 << 24;
    constexpr size_t b_size = sizeof(int) * N;

    // Vectors for holding the host-side (CPU-side) data
    std::vector<int> a(N);
    a.shrink_to_fit();
    std::vector<int> b(N);
    b.shrink_to_fit();
    std::vector<int> c(N);
    c.shrink_to_fit();

    // Initialize random numbers in each array
    generateRandom(a);
    generateRandom(b);

    // Allocate memory on the device (GPU)
    int *d_a, *d_b, *d_c;
    cudaError_t alloc_error = cudaMalloc(&d_a, b_size);
    if (alloc_error != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(alloc_error) << std::endl;
    }
    alloc_error = cudaMalloc(&d_b, b_size);
    if (alloc_error != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(alloc_error) << std::endl;
    }
    alloc_error = cudaMalloc(&d_c, b_size);
    if (alloc_error != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(alloc_error) << std::endl;
    }

    // Copy data from the host to the device (CPU -> GPU)
    cudaError_t memcpy_error = cudaMemcpy(d_a, a.data(), b_size, cudaMemcpyHostToDevice);
    if (memcpy_error != cudaSuccess) {
        std::cout << "Failed to copy data from host to device - error code:" << cudaGetErrorString(memcpy_error) << std::endl;
    }

    memcpy_error = cudaMemcpy(d_b, b.data(), b_size, cudaMemcpyHostToDevice);
    if (memcpy_error != cudaSuccess) {
        std::cout << "Failed to copy data from host to device - error code:" << cudaGetErrorString(memcpy_error) << std::endl;
    }

    // Threads per block (1024)
    int NUM_THREADS = 1 << 10;

    // Blocks number
    int NUM_BLOCKS = static_cast<int>((N + NUM_THREADS - 1) / NUM_THREADS);

    // Launch the kernel on the GPU
    vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    }

    memcpy_error = cudaMemcpy(c.data(), d_c, b_size, cudaMemcpyDeviceToHost);
    if (memcpy_error != cudaSuccess) {
        std::cout << "Failed to copy data from device to host - error code:" << cudaGetErrorString(memcpy_error) << std::endl;
    }

    // Check result for errors
    verifyResult(a, b, c);

    // Show samples
    std::cout << "SOME SAMPLES FROM RESULT VECTOR:" << std::endl;
    showSamples(c);

    // Free memory on device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "\nCOMPLETED SUCCESSFULLY\n";

    return 0;
}