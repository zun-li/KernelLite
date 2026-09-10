#include <cuda_runtime.h>
#include <cstdio>

#define BDIMX 32
#define BDIMY 16

__global__ void transpose(float *out, float *in, int nx, int ny) {
    const int pad = 1;
    __shared__ float tile[BDIMY][BDIMX * 2 + pad];
    
    unsigned int ix = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int ti = iy * nx + ix;

    if (iy < ny) {
        if (ix < nx) tile[threadIdx.y][threadIdx.x] = in[ti];
        if (ix + BDIMX < nx) tile[threadIdx.y][threadIdx.x + BDIMX] = in[ti + BDIMX];
    }
    __syncthreads();

    unsigned int bidx = blockDim.x * threadIdx.y + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    unsigned int ox = blockIdx.y * blockDim.y + icol;
    unsigned int oy = 2 * blockIdx.x * blockDim.x + irow;
    unsigned int to = oy * ny + ox;

    if (ox < ny) {
        if (oy < nx) out[to] = tile[icol][irow];
        if (oy + BDIMX < nx) out[to + ny * BDIMX] = tile[icol][irow + BDIMX];
    }
}

int main() {
    int nx = 4096;
    int ny = 4096;
    size_t size = (size_t)nx * ny * sizeof(float);

    float *h_in = (float *)malloc(size);
    float *h_out = (float *)malloc(size);

    for (int i = 0; i < nx * ny; i++) {
        h_in[i] = float(int(i) % 10);
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 blockSize(BDIMX, BDIMY);
    dim3 gridSize(
        (nx + blockSize.x * 2 - 1) / (blockSize.x * 2),
        (ny + blockSize.y - 1) / blockSize.y
    );
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    transpose<<<gridSize, blockSize>>>(d_out, d_in, nx, ny);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel execution time: %.3f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    long long errors = 0;
    for (int r = 0; r < nx; ++r) {
        for (int c = 0; c < ny; ++c) {
            float expected = h_in[(long long)c * nx + r];
            float got = h_out[(long long)r * ny + c];
            if (got != expected) ++errors;
        }
    }

    if (errors == 0) {
        printf("[PASS] matrix transposition verified.\n");
    } else {
        fprintf(stderr, "[FAIL] %lld mismatches.\n", errors);
    }

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}