# cuda/Makefile
# Compiles the cuBLAS GEMM benchmark with nvcc.
#
# Usage:
#   make          → build  (produces ./cublas_bench)
#   make run      → build + run, redirect CSV output to ../results/cuda_results.csv
#   make clean    → remove build artefacts

NVCC      := nvcc
TARGET    := cublas_bench
SRC       := cublas_benchmark.cu

# -O3          : optimisation level
# -arch=sm_80  : Ampere (A100/RTX 30xx); change to sm_70 for Volta (V100)
#                or sm_86 for GA102 (RTX 3090).  'native' auto-detects on the
#                build machine but may not work on Colab — prefer explicit arch.
NVCC_FLAGS := -O3 -arch=sm_80
LDFLAGS    := -lcublas

RESULTS_DIR := ../results

.PHONY: all build run clean

all: build

build: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LDFLAGS)

run: build
	@mkdir -p $(RESULTS_DIR)
	./$(TARGET) > $(RESULTS_DIR)/cuda_results.csv

clean:
	rm -f $(TARGET) *.o
