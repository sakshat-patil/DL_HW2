"""
PyTorch GEMM Benchmark: CUDA Core (FP32) vs Tensor Core (TF32)
Measures latency and throughput for a single fully-connected (linear) layer
across a sweep of square matrix sizes.
"""

import os
import csv
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SIZES        = [256, 512, 1024, 2048, 4096, 8192]
WARMUP_ITERS = 50
TIMED_ITERS  = 200
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "pytorch_results.csv")


# ---------------------------------------------------------------------------
# Model: single Linear layer, no bias, no activation — pure GEMM
# ---------------------------------------------------------------------------
class FCLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------
def benchmark_gemm(size: int, use_tensor_cores: bool) -> dict:
    """
    Run a GEMM of shape (size x size) @ (size x size) via a Linear layer.
    Returns a dict with latency_ms and tflops.
    """
    M = N = K = size

    # Configure math mode BEFORE constructing model / moving to device
    torch.backends.cuda.matmul.allow_tf32 = use_tensor_cores
    torch.backends.cudnn.allow_tf32       = use_tensor_cores

    device = torch.device("cuda")

    # Build model and input on GPU
    model = FCLayer(K, N).to(device=device, dtype=torch.float32)
    x     = torch.randn(M, K, device=device, dtype=torch.float32)

    # Switch to eval / no_grad for pure compute measurement
    model.eval()

    with torch.no_grad():
        # --- Warm-up ---------------------------------------------------------
        for _ in range(WARMUP_ITERS):
            _ = model(x)
        torch.cuda.synchronize()

        # --- Timed iterations ------------------------------------------------
        start_event = torch.cuda.Event(enable_timing=True)
        end_event   = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(TIMED_ITERS):
            _ = model(x)
        end_event.record()

        # Synchronize so elapsed_time is accurate
        torch.cuda.synchronize()

    total_ms   = start_event.elapsed_time(end_event)   # total time in ms
    latency_ms = total_ms / TIMED_ITERS                 # average per iteration

    # FLOP count: standard GEMM = 2 * M * N * K
    flops      = 2.0 * M * N * K
    tflops     = flops / (latency_ms * 1e-3) / 1e12    # TFLOPS

    return {"latency_ms": latency_ms, "tflops": tflops}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This benchmark requires a GPU.")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"{'Size':>6}  {'Mode':<18}  {'Latency (ms)':>14}  {'Throughput (TFLOPS)':>20}")
    print("-" * 68)

    rows = []
    for size in SIZES:
        for mode_label, use_tc in [("CUDA_Core_FP32", False), ("TensorCore_TF32", True)]:
            result = benchmark_gemm(size, use_tensor_cores=use_tc)
            row = {
                "size":       size,
                "mode":       mode_label,
                "latency_ms": f"{result['latency_ms']:.4f}",
                "tflops":     f"{result['tflops']:.4f}",
            }
            rows.append(row)
            print(
                f"{size:>6}  {mode_label:<18}  "
                f"{result['latency_ms']:>14.4f}  {result['tflops']:>20.4f}"
            )

    # Write CSV
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["size", "mode", "latency_ms", "tflops"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
