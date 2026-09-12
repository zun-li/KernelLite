#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cmath>

// Tile sizes, both must divide seqlen / dim
const int Br = 2;      // Q/O rows handled by one block
const int Bc = 2;      // K/V rows handled by one block
const int seqlen = 4;  // number of query/key rows
const int dim = 4;     // head dimension

// ---------------------------------------------
// Reference (non-flash) attention, one thread per mBlock rows
// ---------------------------------------------

// C = a * A @ B^T + b * C, with A[M, K], B[N, K], C[M, N]
__global__ void naive_sgemm_nt(float *A, float *B, float *C, float a, float b,
                               int M, int N, int K, int mBlock)
{
    int row = (blockDim.x * blockIdx.x + threadIdx.x) * mBlock;

    for (int i = row; i < row + mBlock; i++)
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

// C = a * A @ B + b * C, with A[M, K], B[K, N], C[M, N]
__global__ void naive_sgemm_nn(float *A, float *B, float *C, float a, float b,
                               int M, int N, int K, int mBlock)
{
    int row = (blockDim.x * blockIdx.x + threadIdx.x) * mBlock;

    for (int i = row; i < row + mBlock; i++)
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

// Softmax over each row of a [rows, cols] matrix, one thread per row
__global__ void row_softmax(float *input, float *output, int cols)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    // Row max, for numerical stability
    float maxVal = -INFINITY;
    for (int i = 0; i < cols; i++)
    {
        maxVal = fmaxf(maxVal, input[row * cols + i]);
    }

    // exp(x - max) and its sum
    float sum = 0.f;
    for (int i = 0; i < cols; i++)
    {
        output[row * cols + i] = expf(input[row * cols + i] - maxVal);
        sum += output[row * cols + i];
    }

    // Normalize
    for (int i = 0; i < cols; i++)
    {
        output[row * cols + i] /= sum;
    }
}

// Reference attention: O = softmax(Q @ K^T / sqrt(n)) @ V
// Q, K, V, O are [m, n]; the attention scores are [m, m]
void self_attention_cuda(float *Q, float *K, float *V, float *O, int m, int n)
{
    int mBlock = 2;  // rows computed per thread
    assert(m % mBlock == 0 && "mBlock should align");

    float scale = 1.f / sqrtf(static_cast<float>(n));
    float *scores;
    cudaMalloc(&scores, sizeof(float) * m * m);

    // scores = Q @ K^T
    dim3 qkBlock(m / mBlock);
    naive_sgemm_nt<<<1, qkBlock>>>(Q, K, scores, scale, 0.f, m, m, n, mBlock);
    cudaDeviceSynchronize();

    // scores = softmax(scores), in place
    dim3 smBlock(m);
    row_softmax<<<1, smBlock>>>(scores, scores, m);
    cudaDeviceSynchronize();

    // O = scores @ V
    dim3 pvBlock(m / mBlock);
    naive_sgemm_nn<<<1, pvBlock>>>(scores, V, O, 1.f, 0.f, m, n, m, mBlock);
    cudaDeviceSynchronize();

    cudaFree(scores);
}

// ---------------------------------------------
// FlashAttention v2, one block handles Br query rows
// ---------------------------------------------
__global__ void flash_attention_v2_kernel(float *Q, float *K, float *V, float *O,
                                          int seqlen, float scale)
{
    int kvBlocks = (seqlen + Bc - 1) / Bc;  // number of K/V row blocks
    int dimTilesX = (dim + Bc - 1) / Bc;    // dim tiles owned by tx
    int dimTilesY = (dim + Br - 1) / Br;    // dim tiles owned by ty

    // Shared tiles loaded from global memory
    __shared__ float sQ[Br][dim];
    __shared__ float sK[Bc][dim];
    __shared__ float sV[Bc][dim];
    __shared__ float sO[Br][dim];      // running numerator, unnormalized O
    __shared__ float sScores[Br][Bc];  // Q @ K^T
    __shared__ float sExp[Br][Bc];     // exp(scores - rowMax)
    __shared__ float sRowMax[Br];      // running row max
    __shared__ float sRowSum[Br];      // running sum of exp

    int tx = threadIdx.x;             // indexes Bc / dim tiles
    int ty = threadIdx.y;             // indexes Br
    int row = blockIdx.y * Br + ty;   // query row handled by this thread

    if (row >= seqlen)
    {
        return;
    }

    // Load this thread's Q row, init O and the running statistics
    for (int tile = 0; tile < dimTilesX; tile++)
    {
        sQ[ty][tile * Bc + tx] = Q[row * dim + tile * Bc + tx];
        sO[ty][tile * Bc + tx] = 0.f;
    }
    sRowMax[ty] = -INFINITY;
    sRowSum[ty] = 0.f;

    // Loop over K/V row blocks
    for (int kv = 0; kv < kvBlocks; kv++)
    {
        // Load one block of K and V
        if (kv * Bc + tx < seqlen)
        {
            for (int tile = 0; tile < dimTilesY; tile++)
            {
                sK[tx][tile * Br + ty] = K[(kv * Bc + tx) * dim + tile * Br + ty];
                sV[tx][tile * Br + ty] = V[(kv * Bc + tx) * dim + tile * Br + ty];
            }
        }
        __syncthreads();

        // One score per thread: sScores[ty][tx] = sQ[ty] . sK[tx]
        float score = 0.f;
        for (int k = 0; k < dim; k++)
        {
            score += sQ[ty][k] * sK[tx][k];
        }
        sScores[ty][tx] = score * scale;
        __syncthreads();

        // New row max and exp(scores - rowMax)
        float tileMax = -INFINITY;
        for (int k = 0; k < Bc; k++)
        {
            tileMax = fmaxf(tileMax, sScores[ty][k]);
        }
        float rowMax = fmaxf(sRowMax[ty], tileMax);
        sExp[ty][tx] = expf(sScores[ty][tx] - rowMax);
        __syncthreads();

        // Sum of exp over this block
        float tileSum = 0.f;
        for (int k = 0; k < Bc; k++)
        {
            tileSum += sExp[ty][k];
        }

        // Rescale old O, then add exp @ V for this block
        float rescale = expf(sRowMax[ty] - rowMax);
        for (int tile = 0; tile < dimTilesX; tile++)
        {
            sO[ty][tile * Bc + tx] *= rescale;
            for (int k = 0; k < Bc; k++)
            {
                sO[ty][tile * Bc + tx] += sExp[ty][k] * sV[k][tile * Bc + tx];
            }
        }

        // Update running statistics
        sRowMax[ty] = rowMax;
        sRowSum[ty] = sRowSum[ty] * rescale + tileSum;
        __syncthreads();
    }

    // Normalize the numerator by the denominator and write O
    for (int tile = 0; tile < dimTilesX; tile++)
    {
        O[row * dim + tile * Bc + tx] = sO[ty][tile * Bc + tx] / sRowSum[ty];
    }
}

void flash_attention_v2_cuda(float *Q, float *K, float *V, float *O, int m, int n)
{
    float scale = 1.f / sqrtf(static_cast<float>(n));

    dim3 grid(1, (m + Br - 1) / Br);  // one block per Br query rows
    dim3 block(Bc, Br);               // tx over Bc, ty over Br
    flash_attention_v2_kernel<<<grid, block>>>(Q, K, V, O, m, scale);
}

// Return true if ref and out differ by at most 1e-3 everywhere
bool all_close(float *ref, float *out, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        if (fabs(ref[i] - out[i]) > 1e-3f)
        {
            printf("ref[%d] = %f, out[%d] = %f\n", i, ref[i], i, out[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    const int m = seqlen;  // rows
    const int n = dim;     // columns
    const int size = m * n;

    // Host memory, random Q, K, V
    float *hQ = new float[size];
    float *hK = new float[size];
    float *hV = new float[size];
    float *hRef = new float[size];  // reference output
    float *hOut = new float[size];  // flash output
    for (int i = 0; i < size; i++)
    {
        hQ[i] = static_cast<float>(rand()) / RAND_MAX;
        hK[i] = static_cast<float>(rand()) / RAND_MAX;
        hV[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device memory
    float *dQ, *dK, *dV, *dRef, *dOut;
    cudaMalloc(&dQ, sizeof(float) * size);
    cudaMalloc(&dK, sizeof(float) * size);
    cudaMalloc(&dV, sizeof(float) * size);
    cudaMalloc(&dRef, sizeof(float) * size);
    cudaMalloc(&dOut, sizeof(float) * size);
    cudaMemcpy(dQ, hQ, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, sizeof(float) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, sizeof(float) * size, cudaMemcpyHostToDevice);

    // Time the kernels with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Run both implementations
    self_attention_cuda(dQ, dK, dV, dRef, m, n);
    flash_attention_v2_cuda(dQ, dK, dV, dOut, m, n);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Time for kernel execution: %.3f ms\n", ms);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Compare results
    cudaMemcpy(hRef, dRef, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(hOut, dOut, sizeof(float) * size, cudaMemcpyDeviceToHost);
    printf(all_close(hRef, hOut, m, n) ? "Is equal\n" : "Is not equal\n");

    // Cleanup
    delete[] hQ;
    delete[] hK;
    delete[] hV;
    delete[] hRef;
    delete[] hOut;
    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dRef);
    cudaFree(dOut);

    return 0;
}
