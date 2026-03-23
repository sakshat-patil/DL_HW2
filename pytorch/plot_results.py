"""
Plot GEMM benchmark results: PyTorch and cuBLAS
Generates grouped bar charts for latency and throughput.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (works on headless servers)
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
PLOTS_DIR    = os.path.join(BASE_DIR, "plots")

PYTORCH_CSV  = os.path.join(RESULTS_DIR, "pytorch_results.csv")
CUDA_CSV     = os.path.join(RESULTS_DIR, "cuda_results.csv")

# Labels that appear in the CSV 'mode' column
FP32_LABELS  = {"CUDA_Core_FP32", "cuBLAS_FP32"}
TF32_LABELS  = {"TensorCore_TF32", "cuBLAS_TF32"}


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------
def load_csv(path: str) -> dict:
    """
    Returns a dict:  { mode_label: { size: {"latency_ms": float, "tflops": float} } }
    """
    data = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = row["mode"]
            size = int(row["size"])
            data.setdefault(mode, {})[size] = {
                "latency_ms": float(row["latency_ms"]),
                "tflops":     float(row["tflops"]),
            }
    return data


# ---------------------------------------------------------------------------
# Generic grouped bar-chart helper
# ---------------------------------------------------------------------------
def grouped_bar_chart(
    sizes,
    values_fp32,
    values_tf32,
    ylabel: str,
    title: str,
    out_path: str,
    fp32_label="FP32 (CUDA Cores)",
    tf32_label="TF32 (Tensor Cores)",
):
    """
    Draw a grouped bar chart with two series (FP32 and TF32) grouped by matrix size.
    """
    x          = np.arange(len(sizes))
    bar_width  = 0.35
    fig, ax    = plt.subplots(figsize=(10, 6))

    bars_fp32  = ax.bar(x - bar_width / 2, values_fp32, bar_width,
                        label=fp32_label, color="#4C72B0", edgecolor="white")
    bars_tf32  = ax.bar(x + bar_width / 2, values_tf32, bar_width,
                        label=tf32_label,  color="#DD8452", edgecolor="white")

    # Value annotations on top of each bar
    for bar in bars_fp32:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)
    for bar in bars_tf32:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Matrix Size (N×N)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Per-source plot generation
# ---------------------------------------------------------------------------
def plot_source(data: dict, source_name: str, fp32_label: str, tf32_label: str):
    """Generate latency and throughput plots for one benchmark source."""

    # Identify mode keys present in the data
    fp32_key = next((k for k in data if k in FP32_LABELS), None)
    tf32_key = next((k for k in data if k in TF32_LABELS), None)

    if fp32_key is None or tf32_key is None:
        print(f"  [WARN] Could not find both FP32 and TF32 modes in {source_name} data. "
              f"Keys found: {list(data.keys())}")
        return

    sizes = sorted(data[fp32_key].keys())

    lat_fp32  = [data[fp32_key][s]["latency_ms"] for s in sizes]
    lat_tf32  = [data[tf32_key][s]["latency_ms"] for s in sizes]
    tflops_fp32 = [data[fp32_key][s]["tflops"]   for s in sizes]
    tflops_tf32 = [data[tf32_key][s]["tflops"]   for s in sizes]

    src_lower = source_name.lower()

    # --- Latency plot -------------------------------------------------------
    grouped_bar_chart(
        sizes=sizes,
        values_fp32=lat_fp32,
        values_tf32=lat_tf32,
        ylabel="Average Latency (ms)",
        title=f"{source_name}: FP32 vs TF32 — GEMM Latency",
        out_path=os.path.join(PLOTS_DIR, f"latency_{src_lower}.png"),
        fp32_label=fp32_label,
        tf32_label=tf32_label,
    )

    # --- Throughput plot ----------------------------------------------------
    grouped_bar_chart(
        sizes=sizes,
        values_fp32=tflops_fp32,
        values_tf32=tflops_tf32,
        ylabel="Throughput (TFLOPS)",
        title=f"{source_name}: FP32 vs TF32 — GEMM Throughput",
        out_path=os.path.join(PLOTS_DIR, f"throughput_{src_lower}.png"),
        fp32_label=fp32_label,
        tf32_label=tf32_label,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    missing = []
    for path, label in [(PYTORCH_CSV, "pytorch"), (CUDA_CSV, "cuda")]:
        if not os.path.exists(path):
            missing.append(label)

    if missing:
        print(f"[ERROR] Missing result files for: {', '.join(missing)}")
        print("Run the benchmarks first, then re-run this script.")
        return

    print("Generating PyTorch plots …")
    pytorch_data = load_csv(PYTORCH_CSV)
    plot_source(pytorch_data,
                source_name="PyTorch",
                fp32_label="FP32 (CUDA Cores)",
                tf32_label="TF32 (Tensor Cores)")

    print("Generating cuBLAS plots …")
    cuda_data = load_csv(CUDA_CSV)
    plot_source(cuda_data,
                source_name="CUDA",
                fp32_label="cuBLAS FP32 (CUDA Cores)",
                tf32_label="cuBLAS TF32 (Tensor Cores)")

    print("\nAll plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
