#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int kSeqLen = 1024;
constexpr int kHeadDim = 128;
constexpr int kWarmup = 10;
constexpr int kRepeat = 100;

// scores = Q @ K^T / sqrt(head_dim)
__global__ void qk_kernel(const float* q, const float* k, float* scores,
                          float scale) {
    const int key = blockIdx.x * blockDim.x + threadIdx.x;
    const int query = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;
    for (int d = 0; d < kHeadDim; ++d) {
        sum += q[query * kHeadDim + d] * k[key * kHeadDim + d];
    }
    scores[query * kSeqLen + key] = sum * scale;
}

// In-place softmax, one block per row.
__global__ void softmax_kernel(float* scores) {
    __shared__ float reduction[256];
    float* row = scores + blockIdx.x * kSeqLen;

    float local_max = -INFINITY;
    for (int col = threadIdx.x; col < kSeqLen; col += blockDim.x) {
        local_max = fmaxf(local_max, row[col]);
    }
    reduction[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] =
                fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = reduction[0];

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < kSeqLen; col += blockDim.x) {
        const float value = __expf(row[col] - row_max);
        row[col] = value;
        local_sum += value;
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inverse_sum = 1.0f / reduction[0];
    for (int col = threadIdx.x; col < kSeqLen; col += blockDim.x) {
        row[col] *= inverse_sum;
    }
}

// O = softmax(scores) @ V
__global__ void pv_kernel(const float* scores, const float* v, float* output) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int query = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;
    for (int key = 0; key < kSeqLen; ++key) {
        sum += scores[query * kSeqLen + key] * v[key * kHeadDim + d];
    }
    output[query * kHeadDim + d] = sum;
}

int main() {
    constexpr int tensor_elements = kSeqLen * kHeadDim;
    constexpr int score_elements = kSeqLen * kSeqLen;
    constexpr int tensor_bytes = tensor_elements * sizeof(float);
    constexpr int score_bytes = score_elements * sizeof(float);
    constexpr int peak_bytes = 4 * tensor_bytes + score_bytes;

    std::vector<float> h_q(tensor_elements);
    std::vector<float> h_k(tensor_elements);
    std::vector<float> h_v(tensor_elements);
    std::vector<float> h_output(tensor_elements);
    
    std::srand(1);
    for (float& value : h_q) {
        value = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 0.2f;
    }
    std::srand(2);
    for (float& value : h_k) {
        value = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 0.2f;
    }
    std::srand(3);
    for (float& value : h_v) {
        value = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 0.2f;
    }

    float *d_q, *d_k, *d_v, *d_scores, *d_output;
    cudaMalloc(&d_q, tensor_bytes);
    cudaMalloc(&d_k, tensor_bytes);
    cudaMalloc(&d_v, tensor_bytes);
    cudaMalloc(&d_scores, score_bytes);
    cudaMalloc(&d_output, tensor_bytes);
    cudaMemcpy(d_q, h_q.data(), tensor_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), tensor_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), tensor_bytes, cudaMemcpyHostToDevice);

    const float scale = 1.0f / std::sqrt(kHeadDim);
    const dim3 block(16, 16);
    const dim3 qk_grid(kSeqLen / block.x, kSeqLen / block.y);
    const dim3 pv_grid(kHeadDim / block.x, kSeqLen / block.y);

    for (int i = 0; i < kWarmup; ++i) {
        qk_kernel<<<qk_grid, block>>>(d_q, d_k, d_scores, scale);
        softmax_kernel<<<kSeqLen, 256>>>(d_scores);
        pv_kernel<<<pv_grid, block>>>(d_scores, d_v, d_output);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < kRepeat; ++i) {
        qk_kernel<<<qk_grid, block>>>(d_q, d_k, d_scores, scale);
        softmax_kernel<<<kSeqLen, 256>>>(d_scores);
        pv_kernel<<<pv_grid, block>>>(d_scores, d_v, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    const float latency_ms = total_ms / kRepeat;
    cudaMemcpy(h_output.data(), d_output, tensor_bytes, cudaMemcpyDeviceToHost);

    double checksum = 0.0;
    for (float value : h_output) {
        checksum += value;
    }
    std::printf(
        "self_attn | shape=1x1x%dx%d | latency=%.6f ms | peak_memory=%.2f MiB | checksum=%.6f\n",
        kSeqLen, kHeadDim, latency_ms, peak_bytes / (1024.0 * 1024.0), checksum);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_scores);
    cudaFree(d_output);
    return 0;
}
