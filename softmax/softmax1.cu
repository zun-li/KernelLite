#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

bool compare_results(const float *cpu, const float *gpu, int N, int C) {
    const float epsilon = 1e-3f;
    for (int i = 0; i < N * C; ++i) {
        if (fabs(cpu[i] - gpu[i]) > epsilon)
            return false;
    }
    return true;
}

void softmax_cpu(float *out, float *inp, int N, int C) {
    for (int i = 0; i < N; i ++) {
        const float *inp_row = inp + i * C;
        float *out_row = out + i * C;

        float maxval = -INFINITY;
        for (int j = 0; j < C; j ++) {
            maxval = fmax(maxval, inp_row[j]);
        }

        float sum = 0.f;
        for (int j = 0; j < C; j ++) {
            out_row[j] = expf(inp_row[j] - maxval);
            sum += out_row[j];
        }

        float norm = 1.f / sum;
        for (int j = 0; j < C; j ++) {
            out_row[j] *= norm;
        }
    }
}

template<const int BLOCK_SIZE>
__global__ void softmax_gpu(float *out, float *inp, int N, int C) {
    __shared__ float shared[BLOCK_SIZE];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    const float *inp_row = inp + bid * C;
    float *out_row = out + bid * C;

    // thread coarsening
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += BLOCK_SIZE) {
        maxval = fmax(maxval, inp_row[i]);
    }
    shared[tid] = maxval;
    __syncthreads();

    // reductions
    for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] = fmax(shared[tid], shared[tid + stride]);
        }
    }
    __syncthreads();

    // compute expf and write the result to global memory
    float offset = shared[0];
    for (int i = tid; i < C; i += BLOCK_SIZE) {
        out_row[i] = expf(inp_row[i] - offset);
    }
    __syncthreads();

    // thread coarsening again, for the sum
    float sumval = 0.f;
    for (int i = tid; i < C; i += BLOCK_SIZE) {
        sumval += out_row[i];
    }
    shared[tid] = sumval;
    __syncthreads();

    // reductions
    for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    __syncthreads();

    // divide the input values by the sum
    float sum = shared[0];
    for (int i = tid; i < C; i += BLOCK_SIZE) {
        out_row[i] = out_row[i] / sum;
    }
}

int main() {
    // Example: batch size N=512, classes C=4096
    const int N = 512;
    const int C = 4096;
    const int BLOCK_SIZE = 128;

    size_t num_elements = N * C;
    float *inp = (float *)malloc(num_elements * sizeof(float));
    float *out_cpu = (float *)malloc(num_elements * sizeof(float));
    float *out_gpu = (float *)malloc(num_elements * sizeof(float));

    // Initialize input with sample data
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
        inp[n * C + c] = float(c);
        }
    }

    // Run CPU version and measure time
    auto start_cpu = std::chrono::high_resolution_clock::now();
    softmax_cpu(out_cpu, inp, N, C);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time_ms = end_cpu - start_cpu;

    // Run GPU version and measure time using CUDA events
    float *d_out, *d_inp;
    cudaMalloc((void **)&d_out, N * C * sizeof(float));
    cudaMalloc((void **)&d_inp, N * C * sizeof(float));
    cudaMemcpy(d_inp, inp, N * C * sizeof(float), cudaMemcpyHostToDevice);
    
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    softmax_gpu<BLOCK_SIZE><<<N, BLOCK_SIZE>>>(d_out, d_inp, N, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    cudaMemcpy(out_gpu, d_out, N * C * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare results
    bool success = compare_results(out_cpu, out_gpu, N, C);
    std::cout << "Results match: " << (success ? "YES" : "NO") << std::endl;

    // Print performance comparison
    std::cout << "CPU time: " << cpu_time_ms.count() << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_time_ms << " ms" << std::endl;
    std::cout << "Speedup: " << (cpu_time_ms.count() / (gpu_time_ms)) << "x"
                << std::endl;

    free(inp);
    free(out_cpu);
    free(out_gpu);
    cudaFree(d_out);
    cudaFree(d_inp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}