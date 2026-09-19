#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>

__global__ void hist(uint8_t* input, int* hist, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    for (int idx = i; idx < n; idx += gridDim.x * blockDim.x) {
        uint8_t in = input[idx];
        atomicAdd(&hist[in], 1);
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

    // Histogram warmup
    int warmup_time = 10;
    for (int i = 0; i < warmup_time; i++) {
        hist<<<grid_size, block_size>>>(d_input, d_hist, size);
    }

    cudaDeviceSynchronize();
    cudaMemset(d_hist, 0, 256 * sizeof(int));

    // Histogram
    int repeat_time = 5;
    cudaEventRecord(start);

    for (int i = 0; i < repeat_time; i++) {
        hist<<<grid_size, block_size>>>(d_input, d_hist, size);
    }

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= repeat_time;
    double gelements_per_second = size / (milliseconds * 1e6);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cuda error:%d\n", err);
    }
    printf("Latency: %.6f ms | GElements/s: %.2f\n", milliseconds, gelements_per_second);

    cudaMemset(d_hist, 0, 256 * sizeof(int));
    hist<<<grid_size, block_size>>>(d_input, d_hist, size);

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
