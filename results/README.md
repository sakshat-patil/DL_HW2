# HW1 Excellent — CUDA Core vs Tensor Core GEMM Benchmark

## Overview

This project benchmarks the performance difference between **CUDA Cores (FP32)** and
**Tensor Cores (TF32)** for General Matrix Multiplication (GEMM), the dominant
operation inside fully-connected (linear) layers of deep neural networks.

Two independent implementations are provided:

| Implementation | Backend | File |
|---|---|---|
| PyTorch | `torch.nn.Linear` (no bias, no activation) | `pytorch/benchmark.py` |
| CUDA C++ | cuBLAS `cublasSgemm` / `cublasGemmEx` | `cuda/cublas_benchmark.cu` |

Square matrix sizes swept: **256, 512, 1024, 2048, 4096, 8192**.

Each configuration uses **50 warm-up iterations** followed by **200 timed iterations**,
with GPU events for precise timing.

---

## Hardware / Software Environment

| Item | Value |
|---|---|
| GPU model | NVIDIA A100-SXM4-40GB |
| CUDA version | 13.0 (Driver) / 12.8 (Toolkit) |
| cuBLAS version | Bundled with CUDA 12.8 |
| PyTorch version | 2.10.0+cu128 |
| Python version | 3.11 |
| OS | Google Colab (Ubuntu Linux) |

---

## Project Structure

```
hw1-excellent/
├── README.md
├── Makefile                   # top-level orchestration
├── .gitignore
├── pytorch/
│   ├── benchmark.py           # PyTorch FP32 vs TF32 benchmark
│   └── plot_results.py        # generates all 4 plots
├── cuda/
│   ├── cublas_benchmark.cu    # cuBLAS FP32 vs TF32 benchmark
│   └── Makefile               # nvcc build rules
├── plots/                     # generated PNG files
└── results/                   # CSV output files
```

---

## How to Run

### Prerequisites

- NVIDIA GPU with Tensor Core support (Ampere or newer recommended for TF32)
- CUDA Toolkit ≥ 11.0
- PyTorch ≥ 1.7 with CUDA support
- Python 3.8+, `matplotlib`, `numpy`

```bash
pip install torch matplotlib numpy
```

### Option A — Run everything with the top-level Makefile

```bash
make all
```

### Option B — Run individually

#### PyTorch benchmark

```bash
python3 pytorch/benchmark.py
# Results saved to results/pytorch_results.csv
```

#### CUDA cuBLAS benchmark

```bash
cd cuda
make run        # builds cublas_bench and redirects stdout → ../results/cuda_results.csv
cd ..
```

> **Architecture note:** The `cuda/Makefile` defaults to `-arch=sm_80` (Ampere).
> Change this to match your GPU:
> - Volta V100 → `sm_70`
> - Turing T4/RTX 20xx → `sm_75`
> - Ampere A100/RTX 30xx → `sm_80` / `sm_86`

#### Generate plots

```bash
python3 pytorch/plot_results.py
# Saves 4 PNGs to plots/
```

---

## Results

### Latency (ms) — PyTorch

![PyTorch Latency](plots/latency_pytorch.png)

### Latency (ms) — cuBLAS

![cuBLAS Latency](plots/latency_cuda.png)

### Throughput (TFLOPS) — PyTorch

![PyTorch Throughput](plots/throughput_pytorch.png)

### Throughput (TFLOPS) — cuBLAS

![cuBLAS Throughput](plots/throughput_cuda.png)

### Summary Table — PyTorch

| Matrix Size | FP32 Latency (ms) | TF32 Latency (ms) | FP32 TFLOPS | TF32 TFLOPS | Speedup |
|:-----------:|:-----------------:|:-----------------:|:-----------:|:-----------:|:-------:|
| 256         | 0.03              | 0.03              | 1.11        | 1.05        | 0.9x    |
| 512         | 0.03              | 0.03              | 8.07        | 10.05       | 1.2x    |
| 1024        | 0.14              | 0.03              | 15.31       | 64.25       | 4.2x    |
| 2048        | 1.02              | 0.17              | 16.76       | 101.83      | 6.1x    |
| 4096        | 7.23              | 1.09              | 19.00       | 126.14      | 6.6x    |
| 8192        | 58.03             | 8.44              | 18.95       | 130.23      | 6.9x    |

### Summary Table — cuBLAS

| Matrix Size | FP32 Latency (ms) | TF32 Latency (ms) | FP32 TFLOPS | TF32 TFLOPS | Speedup |
|:-----------:|:-----------------:|:-----------------:|:-----------:|:-----------:|:-------:|
| 256         | 0.01              | 0.01              | 2.29        | 2.61        | 1.1x    |
| 512         | 0.03              | 0.01              | 8.01        | 20.49       | 2.6x    |
| 1024        | 0.16              | 0.04              | 13.58       | 55.04       | 4.1x    |
| 2048        | 1.08              | 0.17              | 15.92       | 102.11      | 6.4x    |
| 4096        | 7.22              | 1.02              | 19.04       | 134.97      | 7.1x    |
| 8192        | 57.45             | 7.72              | 19.14       | 142.42      | 7.4x    |

---

## Analysis

### Why Are Tensor Cores (TF32) Faster?

**1. Reduced mantissa precision → faster multiply-accumulate**

Standard IEEE FP32 uses a 23-bit mantissa. TF32 reduces this to **10 bits** while
keeping the full 8-bit exponent, so the dynamic range is unchanged but each
multiply is far cheaper in silicon. The result is still accumulated into a full
FP32 accumulator, so output precision is largely preserved for typical DNN workloads.

**2. Dedicated Matrix Multiply-Accumulate (MMA) hardware**

Starting with Volta (V100), NVIDIA added **Tensor Core units** alongside the
regular CUDA Cores. A single Tensor Core instruction computes a 4×4×4 MMA in one
clock cycle. Because GEMM maps trivially onto a tiling of MMA operations, entire
warp-level tiles can be retired in a fraction of the time CUDA Cores need for the
equivalent scalar FMA sequence.

**3. Throughput scaling with matrix size**

For small matrices (N ≤ 512) the overhead of loading and storing the tiles can
dominate the raw compute time, and the Tensor Core advantage may be modest. As N
grows the computation becomes **compute-bound** rather than memory-bound, and the
Tensor Core throughput advantage compounds. Our results confirm this: at N=256 there
is essentially no speedup, while at N=8192 we see a **6.9x speedup (PyTorch)** and
**7.4x speedup (cuBLAS)**.

**4. cuBLAS automatically selects Tensor Core algorithms**

When `CUBLAS_TF32_TENSOR_OP_MATH` is set (or `allow_tf32=True` in PyTorch), cuBLAS
picks an algorithm that tiles the problem into shapes optimal for the underlying
Tensor Core MMA instructions. The baseline (`CUBLAS_DEFAULT_MATH`) uses an
algorithm targeting CUDA Cores with full FP32 multiplies throughout, which has
roughly **4× lower peak throughput** compared to Tensor Cores on Ampere.

**5. PyTorch vs cuBLAS comparison**

The PyTorch and cuBLAS results are closely aligned, which is expected since PyTorch
uses cuBLAS under the hood for matrix multiplication. The cuBLAS implementation
shows slightly higher throughput (142 vs 130 TFLOPS at N=8192) because it avoids
the Python/PyTorch overhead and calls cuBLAS directly.

**Summary:** TF32 Tensor Cores are faster because they sacrifice a small amount of
mantissa precision in exchange for dedicated, wide matrix-multiply hardware that
processes data in bulk rather than element-by-element. The speedup is most
pronounced for large, square matrices where arithmetic intensity is high.

---

## References

1. NVIDIA Ampere Architecture Whitepaper — https://images.nvidia.com/aio-uploads/Ampere-GA102-GPU-Architecture-Whitepaper-V2.pdf
2. cuBLAS Library User Guide — https://docs.nvidia.com/cuda/cublas/
3. PyTorch TF32 documentation — https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
4. Markidis et al., "NVIDIA Tensor Core Programmability, Performance & Precision" (arXiv:1803.04014)