#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>


// On GPU section
__global__
void initWithNumber(float num, float *vec, int N)
{

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < N; i += stride) {
        vec[i] = num;
    }
}

__global__
void addVectors(float *result, const float *a, const float *b, int N)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < N; i += stride) {
        result[i] = a[i] + b[i];
    }
}

// On CPU section
void verifyResult(float target, float *vector, int N)
{
    for(int i = 0; i < N; i++) {
        if(vector[i] != target) {
            printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
            exit(1);
        }
    }
    printf("Success! All values calculated correctly.\n");
}

int main()
{
    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    const int N = 2<<24;
    size_t size = N * sizeof(float);

    float *a_gpu, *b_gpu, *c_gpu;
    float *c_cpu;

    cudaMalloc(&a_gpu, size);
    cudaMalloc(&b_gpu, size);
    cudaMalloc(&c_gpu, size);
    cudaMallocHost(&c_cpu, size);

    size_t threadsPerBlock = 1024;
    size_t numberOfBlocks = 32 * numberOfSMs;

    cudaError_t addVectorsErr;
    cudaError_t asyncErr;

    const int streamsCount = 3;
    cudaStream_t streams[streamsCount];

    for(int i = 0; i < streamsCount; i++)
        cudaStreamCreate(&streams[i]);

    initWithNumber<<<numberOfBlocks, threadsPerBlock, 0, streams[0]>>>(3,a_gpu, N);
    initWithNumber<<<numberOfBlocks, threadsPerBlock, 0, streams[1]>>>(4,b_gpu, N);

    // adding in default stream
    addVectors<<<numberOfBlocks, threadsPerBlock>>>(c_gpu, a_gpu, b_gpu, N);

    addVectorsErr = cudaGetLastError();
    if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

    // Copy result from gpu to cpu - it will make more sens to do this in non-default stream when use chunks
    cudaMemcpyAsync(c_cpu, c_gpu, size, cudaMemcpyDeviceToHost, streams[0]);

    asyncErr = cudaDeviceSynchronize();
    if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

    verifyResult(7, c_cpu, N);

    // Destroy streams
    for(int i = 0; i < streamsCount; i++)
        cudaStreamDestroy(streams[i]);

    // and free memory
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);
    cudaFree(c_cpu);
}
