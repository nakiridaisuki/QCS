# Single-gate study build entry point.
#
# Built inside single_gate/env/Dockerfile (or the same image converted to SIF);
# host-native builds are not supported.  scripts pass -arch=sm_75 or -arch=sm_90
# explicitly.
CUQUANTUM_ROOT ?= /usr/local/lib/python3.12/dist-packages/cuquantum
CUQUANTUM_LIB  ?= $(CUQUANTUM_ROOT)/lib
CUDA_HOME      ?= /usr/local/cuda
PYTHON         ?= /opt/single-gate/venv/bin/python
NVCC ?= $(CUDA_HOME)/bin/nvcc
ARCH ?= -gencode "arch=compute_75,code=[sm_75,compute_75]"
# Accept ARCH=sm_75 as well as ARCH="-arch=sm_75" / the default -gencode form.
ifneq (,$(filter -arch=% -gencode%,$(ARCH)))
  ARCH_FLAG := $(ARCH)
else
  ARCH_FLAG := -arch=$(ARCH)
endif

NVCCFLAGS := -O3 -std=c++17 $(ARCH_FLAG) -Xcompiler -fopenmp -I$(CUQUANTUM_ROOT)/include
SVLIB     := -L$(CUQUANTUM_LIB) -l:libcustatevec.so.1 -lcudart \
             -Xlinker -rpath -Xlinker $(CUQUANTUM_LIB)

# Benchmark circuits: downloaded from MQT Bench on demand (needs curl + network;
# run this on the host before entering a container without internet).
# Supported keys live in scripts/baseline/mqt_download.sh.
BENCHMARKS := $(addprefix benchmarks/,graph_state_n25.qasm grover_n17.qasm \
              qaoa_n16.qasm qft_n20.qasm qft_n23.qasm qft_n25.qasm qft_n30.qasm \
              vqe_n16.qasm)

.PHONY: benchmarks
benchmarks: $(BENCHMARKS)

benchmarks/%.qasm:
	scripts/baseline/mqt_download.sh $*

# All single-gate targets (binaries, `make single <table>`, container dispatch)
# live in single_gate/Makefile.
include single_gate/Makefile

.PHONY: clean
clean:
	rm -rf single_gate/build
