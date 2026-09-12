#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <fstream>

__global__ void naive_sgemm_nt(float *A, float *B, float *C,
                               float a, float b, int M, int N, int K, int mBlock)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx *= mBlock;

    for (int i = idx; i < idx + mBlock; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.f;
            for (int k = 0; k < K; k++)
            {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = a * sum + b * C[i * N + j];
        }
    }
}

__global__ void naive_sgemm_nn(float *A, float *B, float *C,
                               float a, float b, int M, int N, int K, int mBlock)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    idx *= mBlock;

    for (int i = idx; i < idx + mBlock; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.f;
            for (int k = 0; k < K; k++)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = a * sum + b * C[i * N + j];
        }
    }
}

__global__ void row_softmax(float *input, float *output, int n)
{
    const float *inp_row = input + blockIdx.x * n;
    float *out_row = output + blockIdx.x * n;

    __shared__ float maxvals[256];
    __shared__ float sumvals[256];

    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        local_max = fmaxf(local_max, inp_row[i]);
    }
    maxvals[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            maxvals[threadIdx.x] = fmaxf(maxvals[threadIdx.x], maxvals[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float max_val = maxvals[0];

    float local_sum = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_sum += __expf(inp_row[i] - max_val);
    }
    sumvals[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sumvals[threadIdx.x] += sumvals[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float sum_val = sumvals[0];

    float inv_sum = 1.f / sum_val;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out_row[i] = __expf(inp_row[i] - max_val) * inv_sum;
    }
}

void self_attention_cuda(float *Q, float *K, float *V, float *O, int m, int n) {
    int rows = 2;  // rows computed per thread
    assert(m % rows == 0 && "rows should align");

    float scale = 1.f / sqrtf(static_cast<float>(n));
    float *scores;  // scores[M, M]
    cudaMalloc(&scores, sizeof(float) * m * m);

    // scores = Q @ K^T
    dim3 qk_blk(m / rows, 1, 1);
    naive_sgemm_nt<<<1, qk_blk>>>(Q, K, scores, scale, 0, m, m, n, rows);
    cudaDeviceSynchronize();

    // softmax each row of scores[M, M]
    dim3 sm_blk(m, 1, 1);
    row_softmax<<<m, sm_blk>>>(scores, scores, m);
    cudaDeviceSynchronize();

    // O = scores[M, M] @ V[M, N]
    dim3 qkv_blk(m / rows, 1, 1);
    naive_sgemm_nn<<<1, qkv_blk>>>(scores, V, O, 1.f, 0.f, m, n, m, rows);
    cudaDeviceSynchronize();

    cudaFree(scores);
}

bool read_bin(const char *filename, float *h_data, size_t num_elements) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        printf("❌ Failed to open %s\n", filename);
        return false;
    }
    file.read((char *)h_data, num_elements * sizeof(float));
    if (!file) {
        printf("❌ Failed to read data from %s\n", filename);
        file.close();
        return false;
    }
    file.close();
    printf("✅ Loaded %s (%zu elements)\n", filename, num_elements);
    return true;
}

bool write_bin(const char *filename, const float *h_data, size_t num_elements) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        printf("❌ Failed to create %s\n", filename);
        return false;
    }
    file.write((const char *)h_data, num_elements * sizeof(float));
    file.close();
    printf("✅ Saved %s (%zu elements)\n", filename, num_elements);
    return true;
}

int main()
{
    const int m = 64;
    const int n = 128;

    printf("🚀 Running self-attention for m=%d, n=%d\n", m, n);

    size_t num_elements = m * n;

    // Host memory
    float *h_Q = new float[num_elements];
    float *h_K = new float[num_elements];
    float *h_V = new float[num_elements];
    float *h_O = new float[num_elements];

    // Read inputs
    read_bin("/home/lz/repo/cuda/flash-attn/data/Q.bin", h_Q, num_elements);
    read_bin("/home/lz/repo/cuda/flash-attn/data/K.bin", h_K, num_elements);
    read_bin("/home/lz/repo/cuda/flash-attn/data/V.bin", h_V, num_elements);

    // Device memory
    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, num_elements * sizeof(float));
    cudaMalloc(&d_K, num_elements * sizeof(float));
    cudaMalloc(&d_V, num_elements * sizeof(float));
    cudaMalloc(&d_O, num_elements * sizeof(float));

    // Copy to device
    cudaMemcpy(d_Q, h_Q, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Run self attention
    self_attention_cuda(d_Q, d_K, d_V, d_O, m, n);

    // Copy result back
    cudaMemcpy(h_O, d_O, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Save output to data/O_cuda.bin
    write_bin("/home/lz/repo/cuda/flash-attn/data/O_cuda.bin", h_O, num_elements);

    // Cleanup
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O;
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);

    printf("🎉 Self-attention completed. Output saved to O_cuda.bin\n");

    return 0;
}