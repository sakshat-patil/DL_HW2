# Top-level Makefile
# Coordinates build, run, and plot targets for the CUDA/PyTorch GEMM benchmark.
#
# Targets:
#   make build-cuda    → compile the CUDA benchmark
#   make run-pytorch   → run the PyTorch benchmark (writes results/pytorch_results.csv)
#   make run-cuda      → compile + run the CUDA benchmark (writes results/cuda_results.csv)
#   make plot          → generate all plots in plots/
#   make all           → build-cuda + run-pytorch + run-cuda + plot

RESULTS_DIR := results
PLOTS_DIR   := plots
CUDA_DIR    := cuda
PY_DIR      := pytorch

.PHONY: all build-cuda run-pytorch run-cuda plot

all: build-cuda run-pytorch run-cuda plot

## Build the CUDA C++ benchmark binary
build-cuda:
	@echo "==> Building CUDA benchmark …"
	$(MAKE) -C $(CUDA_DIR) build

## Run the PyTorch benchmark and save CSV
run-pytorch:
	@echo "==> Running PyTorch benchmark …"
	@mkdir -p $(RESULTS_DIR)
	python3 $(PY_DIR)/benchmark.py

## Compile + run the CUDA benchmark and save CSV
run-cuda:
	@echo "==> Running CUDA benchmark …"
	@mkdir -p $(RESULTS_DIR)
	$(MAKE) -C $(CUDA_DIR) run

## Generate all four plots from the result CSVs
plot:
	@echo "==> Generating plots …"
	@mkdir -p $(PLOTS_DIR)
	python3 $(PY_DIR)/plot_results.py
