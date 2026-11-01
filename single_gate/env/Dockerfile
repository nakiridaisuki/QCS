# Single-gate runtime image: CUDA-Q 0.16 (cu13) + nvcc + nsys/ncu + Python venv.
#
# Build from the repo root:
#   docker build -f single_gate/env/Dockerfile -t single-gate:cudaq0.16 .
#
# This follows route B: start from NVIDIA's tested CUDA-Q image and only add the
# missing build/profiling tools, instead of installing CUDA-Q on a CUDA base.
FROM nvcr.io/nvidia/quantum/cuda-quantum:cu13-0.16.0

USER root
ENV DEBIAN_FRONTEND=noninteractive

# The image already has cuda-cudart-13-0 13.0.96; only nvcc + build tools are
# missing.  nsys/ncu are pinned to the versions used for the report.
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvcc-13-0=13.0.88-1 \
        build-essential \
        python3-venv \
        nsight-systems-2025.3.2=2025.3.2.474-253236389321v0 \
        nsight-compute-2025.3.0=2025.3.0.19-1 \
    && ln -sf /opt/nvidia/nsight-compute/2025.3.0/ncu /usr/local/bin/ncu \
    && rm -rf /var/lib/apt/lists/*

# system-site-packages keeps CUDA-Q (and cuQuantum) from the base image visible
# while qiskit/numpy/scipy live in the venv.
RUN python3 -m venv --system-site-packages /opt/single-gate/venv \
    && /opt/single-gate/venv/bin/pip install --no-cache-dir \
        qiskit==2.6.0 numpy==2.5.4 scipy==1.18.1

ENV CUDA_HOME=/usr/local/cuda
ENV CUQUANTUM_ROOT=/usr/local/lib/python3.12/dist-packages/cuquantum
ENV CUQUANTUM_LIB=/usr/local/lib/python3.12/dist-packages/cuquantum/lib
ENV CUDAQ_VENV=/opt/single-gate/venv
ENV PY=/opt/single-gate/venv/bin/python
ENV PATH=/opt/single-gate/venv/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib/python3.12/dist-packages/cuquantum/lib

WORKDIR /workspace
