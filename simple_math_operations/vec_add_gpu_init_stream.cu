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

    float *a, *b;
    float *c;

    cudaError_t addVectorsErr;
    cudaError_t asyncErr;

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    // inform GPU it will need those (works only for Linux)
//    cudaMemPrefetchAsync(a, size, deviceId);
//    cudaMemPrefetchAsync(b, size, deviceId);
//    cudaMemPrefetchAsync(c, size, deviceId);

    addVectorsErr = cudaGetLastError();
    if(addVectorsErr != cudaSuccess) printf("Error when prefetch: %s\n", cudaGetErrorString(addVectorsErr));

    size_t threadsPerBlock = 1024;
    size_t numberOfBlocks = 32 * numberOfSMs;

    cudaStream_t streamA;
    cudaStream_t streamB;
    cudaStream_t streamC;

    cudaStreamCreate(&streamA);
    cudaStreamCreate(&streamB);
    cudaStreamCreate(&streamC);

    initWithNumber<<<numberOfBlocks, threadsPerBlock, 0, streamA>>>(3, a, N);
    initWithNumber<<<numberOfBlocks, threadsPerBlock, 0, streamB>>>(4, b, N);
    initWithNumber<<<numberOfBlocks, threadsPerBlock, 0, streamC>>>(0, c, N);

    addVectors<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

    addVectorsErr = cudaGetLastError();
    if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

    asyncErr = cudaDeviceSynchronize();
    if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

    cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);

    verifyResult(7, c, N);

    // Do not forget to destroy streams
    cudaStreamDestroy(streamA);
    cudaStreamDestroy(streamB);
    cudaStreamDestroy(streamC);

    // and free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
