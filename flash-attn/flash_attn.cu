#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int kSeqLen = 1024;
constexpr int kHeadDim = 128;
constexpr int kQueryRows = 4;
constexpr int kKeyRows = 32;
constexpr int kWarmup = 10;
constexpr int kRepeat = 100;

__global__ void flash_attention_kernel(const float* q, const float* k,
                                       const float* v, float* output,
                                       float scale) {
    __shared__ float q_tile[kQueryRows][kHeadDim];
    __shared__ float k_tile[kKeyRows][kHeadDim];
    __shared__ float v_tile[kKeyRows][kHeadDim];
    __shared__ float output_tile[kQueryRows][kHeadDim];
    // 先保存当前 tile 的 score，随后原地改为未归一化的 softmax 权重。
    __shared__ float scores[kQueryRows][kKeyRows];
    __shared__ float row_max[kQueryRows];
    __shared__ float row_sum[kQueryRows];
    __shared__ float rescale[kQueryRows];

    const int key_in_tile = threadIdx.x;
    const int query_in_tile = threadIdx.y;
    const int query = blockIdx.x * kQueryRows + query_in_tile;
    const int linear_thread = query_in_tile * kKeyRows + key_in_tile;
    constexpr int threads_per_block = kQueryRows * kKeyRows;

    // 每个 block 常驻 kQueryRows 行 Q，并维护对应的输出分子。
    for (int d = key_in_tile; d < kHeadDim; d += kKeyRows) {
        q_tile[query_in_tile][d] = q[query * kHeadDim + d];
        output_tile[query_in_tile][d] = 0.0f;
    }
    if (key_in_tile == 0) {
        row_max[query_in_tile] = -INFINITY;
        row_sum[query_in_tile] = 0.0f;
    }
    __syncthreads();

    // K、V 每次只读入 kKeyRows 行，因此无需保存完整 attention 矩阵。
    for (int key_start = 0; key_start < kSeqLen; key_start += kKeyRows) {
        constexpr int tile_elements = kKeyRows * kHeadDim;
        for (int index = linear_thread; index < tile_elements;
             index += threads_per_block) {
            const int tile_row = index / kHeadDim;
            const int d = index % kHeadDim;
            k_tile[tile_row][d] =
                k[(key_start + tile_row) * kHeadDim + d];
            v_tile[tile_row][d] =
                v[(key_start + tile_row) * kHeadDim + d];
        }
        __syncthreads();

        float score = 0.0f;
        for (int d = 0; d < kHeadDim; ++d) {
            score += q_tile[query_in_tile][d] * k_tile[key_in_tile][d];
        }
        scores[query_in_tile][key_in_tile] = score * scale;
        __syncthreads();

        // 在线 softmax：合并当前 tile 与之前 tile 的最大值和指数和。
        if (key_in_tile == 0) {
            float tile_max = -INFINITY;
            for (int key = 0; key < kKeyRows; ++key) {
                tile_max = fmaxf(tile_max, scores[query_in_tile][key]);
            }
            const float new_max = fmaxf(row_max[query_in_tile], tile_max);
            const float old_scale = __expf(row_max[query_in_tile] - new_max);
            float tile_sum = 0.0f;
            for (int key = 0; key < kKeyRows; ++key) {
                const float probability =
                    __expf(scores[query_in_tile][key] - new_max);
                scores[query_in_tile][key] = probability;
                tile_sum += probability;
            }
            rescale[query_in_tile] = old_scale;
            row_sum[query_in_tile] =
                row_sum[query_in_tile] * old_scale + tile_sum;
            row_max[query_in_tile] = new_max;
        }
        __syncthreads();

        for (int d = key_in_tile; d < kHeadDim; d += kKeyRows) {
            float value =
                output_tile[query_in_tile][d] * rescale[query_in_tile];
            for (int key = 0; key < kKeyRows; ++key) {
                value += scores[query_in_tile][key] * v_tile[key][d];
            }
            output_tile[query_in_tile][d] = value;
        }
        __syncthreads();
    }

    const float inverse_sum = 1.0f / row_sum[query_in_tile];
    for (int d = key_in_tile; d < kHeadDim; d += kKeyRows) {
        output[query * kHeadDim + d] =
            output_tile[query_in_tile][d] * inverse_sum;
    }
}

int main() {
    constexpr int tensor_elements = kSeqLen * kHeadDim;
    constexpr int tensor_bytes = tensor_elements * sizeof(float);
    constexpr int peak_bytes = 4 * tensor_bytes;
    constexpr int shared_bytes = sizeof(float) * (2 * kQueryRows * kHeadDim + 2 * kKeyRows * kHeadDim + kQueryRows * kKeyRows + 3 * kQueryRows);

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

    float *d_q, *d_k, *d_v, *d_output;
    cudaMalloc(&d_q, tensor_bytes);
    cudaMalloc(&d_k, tensor_bytes);
    cudaMalloc(&d_v, tensor_bytes);
    cudaMalloc(&d_output, tensor_bytes);
    cudaMemcpy(d_q, h_q.data(), tensor_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), tensor_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), tensor_bytes, cudaMemcpyHostToDevice);

    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    const dim3 block(kKeyRows, kQueryRows);
    const dim3 grid(kSeqLen / kQueryRows);

    for (int i = 0; i < kWarmup; ++i) {
        flash_attention_kernel<<<grid, block>>>(d_q, d_k, d_v, d_output, scale);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < kRepeat; ++i) {
        flash_attention_kernel<<<grid, block>>>(d_q, d_k, d_v, d_output, scale);
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
        "flash_attn | shape=1x1x%dx%d | latency=%.6f ms | peak_memory=%.2f MiB | shared_memory_per_block=%.2f KiB | checksum=%.6f\n",
        kSeqLen, kHeadDim, latency_ms, peak_bytes / (1024.0 * 1024.0), shared_bytes / 1024.0, checksum);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_output);
    return 0;
}
