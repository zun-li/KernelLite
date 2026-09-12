// rmsnorm_cuda_test.cpp
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

void row_rmsnorm_f32_dim_cpu(float *in, float *weight, float *out, int batch,
                             int size, float eps) {
    for (int i = 0; i < batch; ++i) {
        float *in_ptr = in + i * size;
        float *out_ptr = out + i * size;

        float sum = 0.0f;
        for (int j = 0; j < size; ++j) {
            float val = in_ptr[j];
            sum += val * val;
        }
        float rms = 1.0f / std::sqrt(sum / static_cast<float>(size) + eps);

        for (int j = 0; j < size; ++j) {
            float x = in_ptr[j] * weight[j];
            out_ptr[j] = x * rms;
        }
    }
}

__inline__ __device__ float block_reduce(float val) {
    const int tid = threadIdx.x;
    const int warpSize = 32;
    int laneId = tid % warpSize;
    int warpId = tid / warpSize;

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    // Write warp result to shared memory
    __shared__ float warpSums[32]; // Max 32 warps per block
    if (laneId == 0) {
        warpSums[warpId] = val;
    }
    __syncthreads();

    // Final reduction: only first warp participates
    if (warpId == 0) {
        val = (tid < (blockDim.x + warpSize - 1) / warpSize) ? warpSums[tid] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    } else {
        val = 0.0f;
    }
    return val;
}

__global__ void row_rmsnorm_f32_dim(float *in, float *wei, float *out,
                                    int batch, int size, float eps)
{
    const int bid = blockIdx.x;
    if (bid >= batch)
        return;

    float *block_in = in + bid * size;
    float *block_out = out + bid * size;
    float sum = 0.0f;

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float x = block_in[i];
        sum += x * x;
    }
    __shared__ float shared_val;
    sum = block_reduce(sum);

    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    const float scale = rsqrtf(sum / static_cast<float>(size) + eps);
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float x = block_in[i] * wei[i];
        block_out[i] = x * scale;
    }
}

float compute_max_error(const std::vector<float> &cpu_out,
                        const std::vector<float> &cuda_out, int n)
{
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        float err = std::abs(cpu_out[i] - cuda_out[i]);
        max_err = std::max(max_err, err);
        if (max_err > 1.f) {
            printf("Error at index %6d: CPU = %14.6f, CUDA = %14.6f, Error = %14.6e\n",
                   i, cpu_out[i], cuda_out[i], err);
            break;
        }
    }
    return max_err;
}

// ----------------------------
// Main Function
// ----------------------------
int main() {
    const int batch = 16;
    const int size = 1024;
    const float eps = 1e-6f;
    const int total = batch * size;

    // Host memory
    std::vector<float> h_input(total);
    std::vector<float> h_weight(size);
    std::vector<float> h_output_cpu(total);
    std::vector<float> h_output_cuda(total);

    // Random init
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < total; ++i) {
        h_input[i] = dis(gen);
    }

    for (int i = 0; i < size; ++i) {
        h_weight[i] = dis(gen);
    }

    // CPU version
    auto start_cpu = std::chrono::high_resolution_clock::now();
    row_rmsnorm_f32_dim_cpu(
        h_input.data(), h_weight.data(), h_output_cpu.data(), batch, size, eps);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
    const double cpu_ms = duration.count() / 1000.0;
    printf("CPU  RMSNorm took %10.3f ms\n", cpu_ms);

    // CUDA setup
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, total * sizeof(float));
    cudaMalloc(&d_weight, size * sizeof(float));
    cudaMalloc(&d_output, total * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch config
    const int block_size = 1024;
    const int grid_size = batch; // One block per batch row
    dim3 grid(grid_size);
    dim3 block(block_size);

    // CUDA timing with events
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // Warm-up run
    int warpup = 10;
    for (int i = 0; i < warpup; i++) {
        row_rmsnorm_f32_dim<<<grid, block>>>
            (d_input, d_weight, d_output, batch, size, eps);
    }
    
    cudaEventRecord(start_gpu);
    int test_iter = 10;
    for (int i = 0; i < test_iter; ++i) {
        row_rmsnorm_f32_dim<<<grid, block>>>
            (d_input, d_weight, d_output, batch, size, eps);
    }
    cudaEventRecord(stop_gpu);

    // kernel 是异步的，必须等 stop 事件真正完成后才能计时；
    // 否则 cudaEventElapsedTime 返回 cudaErrorNotReady 且 cuda_time 未初始化。
    cudaEventSynchronize(stop_gpu);

    float cuda_time;
    cudaEventElapsedTime(&cuda_time, start_gpu, stop_gpu);

    // Copy result back
    cudaMemcpy(h_output_cuda.data(), d_output, total * sizeof(float),
               cudaMemcpyDeviceToHost);

    printf("CUDA RMSNorm took %10.3f ms (avg over %d iters)\n",
           cuda_time / test_iter, test_iter);

    // Compare results
    float max_error = compute_max_error(h_output_cpu, h_output_cuda, total);
    printf("Max absolute error (CPU vs CUDA): %.6e\n", max_error);

    // Optional: print first few values
    printf("\nFirst 10 outputs (CPU vs CUDA):\n");
    printf("%4s %16s %16s %16s\n", "Idx", "CPU", "CUDA", "Diff");
    for (int i = 0; i < 10; ++i)
    {
        printf("%4d %16.6f %16.6f %16.6e\n", i, h_output_cpu[i], h_output_cuda[i],
               std::abs(h_output_cpu[i] - h_output_cuda[i]));
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}
