#include <cuda_runtime.h>
#include <cstdio>

#define BDIMX 32
#define BDIMY 16

__global__ void transpose(float *out, float *in, int nx, int ny) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        out[ix * ny + iy] = in[iy * nx + ix];
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
        (nx + blockSize.x - 1) / blockSize.x,
        (ny + blockSize.y - 1) / blockSize.y
    );
    transpose<<<gridSize, blockSize>>>(d_out, d_in, nx, ny);

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
        printf("PASS: matrix transposition verified.\n");
    } else {
        fprintf(stderr, "FAIL: %lld mismatches.\n", errors);
    }

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}