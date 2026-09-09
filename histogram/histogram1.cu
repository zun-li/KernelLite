#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>

__global__ void hist(uint8_t* input, int* hist, int n) {
    __shared__ int histo_private[256];

    for (int j = threadIdx.x; j < 256; j += blockDim.x) {
        histo_private[j] = 0;
    }
    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (int idx = i; idx < n; idx += gridDim.x * blockDim.x) {
        uint8_t in = input[idx];
        atomicAdd(&histo_private[in], 1);
    }
    __syncthreads();

    for (int j = threadIdx.x; j < 256; j += blockDim.x) {
        atomicAdd(&hist[j], histo_private[j]);
    }
}

int main() {
    int M = 4096;
    int N = 4096;
    int size = M * N;
    uint8_t* input = new uint8_t[size];
    for (int i = 0; i < size; ++i) {
        input[i] = rand() % 256;
    }

    uint8_t* d_input;
    int* d_hist;
    cudaMalloc(&d_input, size * sizeof(uint8_t));
    cudaMalloc(&d_hist, 256 * sizeof(int));
    cudaMemset(d_hist, 0, 256 * sizeof(int));

    dim3 block_size(256);
    dim3 grid_size(256);
    cudaMemcpy(d_input, input, sizeof(uint8_t) * size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    hist<<<grid_size, block_size>>>(d_input, d_hist, size);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cuda error:%d\n", err);
    }
    printf("Kernel execution time: %f ms\n", milliseconds);

    int h_hist[256];
    cudaMemcpy(h_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        printf("%d : %d\n", i, h_hist[i]);
    }

    cudaFree(d_input);
    cudaFree(d_hist);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] input;

    return 0;
}