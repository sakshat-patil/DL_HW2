/*
 * cublas_benchmark.cu
 *
 * Measures GEMM (C = A * B) latency and throughput for:
 *   - Baseline  : cublasSgemm  with CUBLAS_DEFAULT_MATH   (CUDA Cores, FP32)
 *   - TF32 path : cublasGemmEx with CUBLAS_TF32_TENSOR_OP_MATH (Tensor Cores)
 *
 * Square matrices of sizes [256, 512, 1024, 2048, 4096, 8192] are swept.
 * Results are printed as CSV to stdout so they can be redirected:
 *
 *   ./cublas_bench > ../results/cuda_results.csv
 *
 * Timing: cudaEvent_t with 50 warm-up + 200 timed iterations.
 *
 * Compile:  see cuda/Makefile
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ---------------------------------------------------------------------------
// Error-checking macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t status = (call);                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "[CUBLAS ERROR] %s:%d  status=%d\n",               \
                    __FILE__, __LINE__, (int)status);                           \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static const int SIZES[]      = {256, 512, 1024, 2048, 4096, 8192};
static const int NUM_SIZES    = (int)(sizeof(SIZES) / sizeof(SIZES[0]));
static const int WARMUP_ITERS = 50;
static const int TIMED_ITERS  = 200;

// ---------------------------------------------------------------------------
// Matrix initialisation (simple deterministic fill; avoids cuRAND dependency)
// ---------------------------------------------------------------------------
__global__ void fill_matrix_kernel(float* M, int n, float val, int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        // Vary values slightly to prevent degenerate optimisations
        M[idx] = val + 0.01f * (float)(idx % stride);
}

static void fill_matrix(float* d_M, int rows, int cols, float val)
{
    int n = rows * cols;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    fill_matrix_kernel<<<blocks, threads>>>(d_M, n, val, cols);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Benchmark one size in one mode
// ---------------------------------------------------------------------------
struct BenchResult {
    double latency_ms;
    double tflops;
};

/*
 * run_benchmark
 *
 * @param handle        cuBLAS handle (reused across calls)
 * @param N             matrix dimension (square: M=N=K=N)
 * @param use_tc        true  → TF32 Tensor Core path (cublasGemmEx)
 *                      false → FP32 CUDA Core path  (cublasSgemm)
 */
static BenchResult run_benchmark(cublasHandle_t handle, int N, bool use_tc)
{
    size_t bytes = (size_t)N * N * sizeof(float);

    // Allocate device matrices A, B, C
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Initialise with simple values
    fill_matrix(d_A, N, N, 1.0f);
    fill_matrix(d_B, N, N, 0.5f);
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());

    // cuBLAS uses column-major layout; for row-major A*B we compute B^T * A^T
    // (standard trick: swapping A↔B gives the correct result for square matrices)
    const float alpha = 1.0f, beta = 0.0f;

    // Lambda-like helper: one GEMM call
    auto gemm_call = [&]() {
        if (!use_tc) {
            // ----------------------------------------------------------------
            // Baseline: standard SGEMM, math mode explicitly set to DEFAULT
            // (disables Tensor Cores even on Ampere/Volta GPUs)
            // ----------------------------------------------------------------
            CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
            CUBLAS_CHECK(cublasSgemm(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_B, N,   // leading dimension = N (column-major)
                d_A, N,
                &beta,
                d_C, N));
        } else {
            // ----------------------------------------------------------------
            // TF32 Tensor Core path: cublasGemmEx with TF32 math mode.
            // Inputs/output stay in FP32 memory; the compute is TF32 internally.
            // ----------------------------------------------------------------
            CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
            CUBLAS_CHECK(cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_B, CUDA_R_32F, N,
                d_A, CUDA_R_32F, N,
                &beta,
                d_C, CUDA_R_32F, N,
                CUBLAS_COMPUTE_32F_FAST_TF32,   // TF32 compute type
                CUBLAS_GEMM_DEFAULT_TENSOR_OP)); // allow Tensor Core algorithm
        }
    };

    // --- Warm-up ------------------------------------------------------------
    for (int i = 0; i < WARMUP_ITERS; ++i)
        gemm_call();
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Timed iterations ---------------------------------------------------
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < TIMED_ITERS; ++i)
        gemm_call();
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_stop));

    double latency_ms = (double)total_ms / TIMED_ITERS;

    // FLOP count: 2 * M * N * K  (standard GEMM)
    double flops  = 2.0 * (double)N * (double)N * (double)N;
    double tflops = flops / (latency_ms * 1e-3) / 1e12;

    // Clean up
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return {latency_ms, tflops};
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(void)
{
    // Print device name for reference (goes to stderr so it does not corrupt CSV)
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    fprintf(stderr, "Device: %s\n", prop.name);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // CSV header
    printf("size,mode,latency_ms,tflops\n");

    for (int i = 0; i < NUM_SIZES; ++i) {
        int N = SIZES[i];

        // --- Baseline: FP32, CUDA Cores ------------------------------------
        BenchResult fp32 = run_benchmark(handle, N, /*use_tc=*/false);
        printf("%d,cuBLAS_FP32,%.4f,%.4f\n", N, fp32.latency_ms, fp32.tflops);
        fflush(stdout);

        // --- TF32 Tensor Core path ------------------------------------------
        BenchResult tf32 = run_benchmark(handle, N, /*use_tc=*/true);
        printf("%d,cuBLAS_TF32,%.4f,%.4f\n", N, tf32.latency_ms, tf32.tflops);
        fflush(stdout);

        // Progress to stderr
        fprintf(stderr, "  [%4d]  FP32: %8.3f ms / %6.3f TFLOPS  |  "
                         "TF32: %8.3f ms / %6.3f TFLOPS\n",
                N,
                fp32.latency_ms, fp32.tflops,
                tf32.latency_ms, tf32.tflops);
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
