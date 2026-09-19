# KernelLite

KernelLite 是一个用于学习和实践 CUDA Kernel 优化的项目。

项目包含 SGEMM、Softmax、Transpose、Histogram、RMSNorm 和 FlashAttention 等常见计算任务，并通过多个实现版本展示不同 Optimization Strategy 对性能的影响。

## Performance

### Benchmark Environment

| Item | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 |
| NVIDIA Driver | 580.173.02 |
| CUDA | CUDA 13.0 |
| Compiler | nvcc 13.0.48 / GCC 13.3.0 |
| OS | Ubuntu 24.04.5 LTS |
| CPU | 12th Gen Intel Core i5-12490F |

### SGEMM

测试采用 FP32 方阵（`M = N = K`，`N = 128 ~ 8192`）。

每项预热 10 次，随后运行 5 次取平均。

性能按 `2MNK / time` 计算，单位为 GFLOP/s。

#### Implementations

| Version  | Optimization                          |
| -------- | ------------------------------------- |
| `sgemm0` | Naive Implementation                  |
| `sgemm1` | Shared Memory Tiling                  |
| `sgemm2` | Thread Tiling                         |
| `sgemm3` | Register Blocking + Vectorized Access |
| `sgemm4` | Double Buffering                      |
| `cuBLAS` | NVIDIA Optimized Implementation       |

#### Benchmark Results

| N    | `sgemm0` | `sgemm1` | `sgemm2` | `sgemm3` | `sgemm4` | cuBLAS |
| ---: | -------: | -------: | -------: | -------: | -------: | -----: |
|  128 |   499.51 |   607.94 |   195.05 |   217.87 |   280.55 | 930.91 |
|  256 |   810.21 |  1030.44 |   866.88 |   959.88 |  1277.82 | 3723.64 |
|  512 |   923.04 |  1194.82 |  3661.23 |  3996.10 |  5353.98 | 7150.19 |
| 1024 |   891.49 |  1239.74 |  5576.70 |  6704.45 |  7895.90 | 8637.36 |
| 2048 |  1035.34 |  1335.77 |  5823.80 |  7063.66 |  9121.03 | 9340.49 |
| 4096 |   893.54 |  1214.68 |  4875.93 |  7953.93 |  8503.40 | 8241.70 |
| 8192 |   891.93 |  1184.66 |  4807.06 |  7818.83 |  8433.76 | 8367.19 |

cuBLAS 结果取五组运行数据的中位数。

### Softmax

输入规模：`512 × 4096`，FP32。

每项 GPU 测试预热 100 次，随后运行 100 次取平均。

Speedup 按 `CPU time / GPU time` 计算。

| Version    | Optimization            | CPU Time (ms) | GPU Time (ms) | Speedup |
| ---------- | ----------------------- | ------------: | ------------: | ------: |
| `softmax0` | One Thread per Row      |       17.8385 |          1.33 |  13.41× |
| `softmax1` | Shared Memory Reduction |       17.7147 |     0.0477901 | 370.68× |
| `softmax2` | Warp Shuffle Reduction  |       17.5505 |     0.0647578 | 271.02× |
| `softmax3` | Multi-Warp Reduction    |       17.6104 |     0.0551658 | 319.23× |

### Transpose

输入规模：`4096 × 4096`，FP32。

每项预热 10 次，随后运行 5 次取平均。

有效带宽按 `2 × nx × ny × sizeof(float) / latency` 计算，其中包含一次读取和一次写入。

| Version      | Optimization            | Latency (ms) | Effective GB/s | Speedup |
| ------------ | ----------------------- | -----------: | -------------: | ------: |
| `transpose0` | Naive Transpose         |        1.502 |          89.36 |   1.00× |
| `transpose1` | Shared Memory Transpose |        0.703 |         190.92 |   2.14× |
| `transpose2` | Shared Memory Padding   |        0.574 |         233.83 |   2.62× |
| `transpose3` | Thread Coarsening       |        0.684 |         196.22 |   2.20× |

### Histogram

输入规模：`4096 × 4096`，UINT8，256 bins。

每项预热 10 次，随后运行 5 次取平均。

GElements/s 按 `(M × N) / (latency × 10^6)` 计算，其中 latency 的单位为 ms。

| Version      | Optimization                | Latency (ms) | GElements/s | Speedup |
| ------------ | --------------------------- | -----------: | ----------: | ------: |
| `histogram0` | Global Atomic Operations    |     4.434535 |        3.78 |   1.00× |
| `histogram1` | Shared Memory Privatization |     0.060621 |      276.76 |  73.15× |

### RMSNorm

输入规模：`batch = 16, hidden_size = 1024`，FP32。

每项预热 10 次，随后运行 1000 次取平均。

GElements/s 按 `(batch × hidden_size) / (latency × 10^6)` 计算，其中 latency 的单位为 ms。

| Version    | Optimization               | CPU Time (ms) | GPU Latency (ms) | GElements/s | Max Absolute Error |
| ---------- | -------------------------- | ------------: | ---------------: | ----------: | -----------------: |
| `rmsnorm0` | Block Reduction            |         0.059 |         0.001956 |        8.38 |       2.384186e-06 |
| `rmsnorm1` | Vectorized `float4` Access |         0.063 |         0.001961 |        8.35 |       2.861023e-06 |

### FlashAttention

输入规模统一表示为 `batch × heads × sequence_length × head_dim`，FP32。

| Version      | Optimization            | Latency | GFLOP/s | Speedup | Peak Memory |
| ------------ | ----------------------- | ------: | -------: | ------: | ----------: |
| `self_attn`  | Standard Attention      |     TBD |      TBD |   1.00× |         TBD |
| `flash_attn` | Tiling + Online Softmax |     TBD |      TBD |     TBD |         TBD |
