#include <cstdio>
#include <cstdint>
#include <iostream>

__global__ void hist(uint8_t *input, int *hist, int n) {
    __shared__ int local_hist[256];

    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        atomicAdd(&local_hist[input[i]], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        atomicAdd(&hist[i], local_hist[i]);
    }
}

int main() {
    const int M = 4096;
    const int N = 4096;

    uint8_t *input = (uint8_t*)malloc(sizeof(uint8_t) * (M * N));

    for (int i = 0; i < M * N; i ++) {
        input[i] = i % 256;
    }

    uint8_t *d_input;
    int *d_hist;
    cudaMalloc(&d_input, sizeof(uint8_t) * M * N);
    cudaMalloc(&d_hist, sizeof(int) * 256);
    cudaMemset(d_hist, 0, sizeof(int) * 256);
    cudaMemcpy(d_input, input, sizeof(uint8_t) * M * N, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    hist<<<32, 32>>> (d_input, d_hist, M * N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    int hist[256];
    cudaMemcpy(hist, d_hist, sizeof(int) * 256, cudaMemcpyDeviceToHost);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "cost time: " << ms << "ms." << std::endl;

    bool flag = true;
    for (int i = 0; i < 256; i ++) {
        if (hist[i] != 65536) {
            flag = false;
        }
    }

    if (flag) {
        std::cout << "results right!" << std::endl;
    } else {
        std::cout << "results error!" << std::endl;
    }

    free(input);
    cudaFree(d_input);
    cudaFree(d_hist);

    return 0;
}