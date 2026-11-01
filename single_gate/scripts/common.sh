#!/usr/bin/env bash
# Shared bootstrap for single_gate/jobs/*.sh.  Source it, then call sg_init.
#
#   source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
#   sg_init
#   sg_env
#   sg_gpu_info
#
# Everything runs inside single_gate/env/Dockerfile (or the same image converted
# to SIF).  Host-native execution is no longer supported.
# ARCH defaults to sm_75; H200 jobs must pass ARCH=sm_90 explicitly.

: "${ARCH:=sm_75}"
export ARCH

PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PROJ

# Image-internal locations; override only when the image layout changes.
export V=${CUDAQ_VENV:-/opt/single-gate/venv}

sg_init() { # [table-name]  (default: script basename without .sh)
    export NAME=${1:-$(basename "$0" .sh)}
    export RAW="$PROJ/single_gate/results/raw"
    export TABLES="$PROJ/single_gate/results/tables"
    export NSYS_LOGS="$PROJ/single_gate/results/nsys_logs"
    mkdir -p "$RAW" "$TABLES" "$NSYS_LOGS" "$PROJ/single_gate/build"
    cd "$PROJ"
    # Under plain sbatch the script's stdout is already the job log in RAW.
    # Inside Apptainer/Docker it is not, so tee there even when SLURM_JOB_ID is set.
    if [ -z "${SLURM_JOB_ID:-}" ] || [ -n "${APPTAINER_CONTAINER:-}" ] \
            || [ -n "${SINGULARITY_CONTAINER:-}" ] || [ -f /.dockerenv ]; then
        exec > >(tee "$RAW/$NAME-${SLURM_JOB_ID:-local}.out") 2>&1
    fi
}

sg_env() {
    export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    CUQ=${CUQUANTUM_ROOT:-/usr/local/lib/python3.12/dist-packages/cuquantum}
    export CUQUANTUM_ROOT="$CUQ" CUQUANTUM_LIB="$CUQ/lib"
    export LD_LIBRARY_PATH="$CUQ/lib:$LD_LIBRARY_PATH"
    export CUSTATEVEC_LIB='-l:libcustatevec.so.1'
    export NVCCFLAGS="-O3 -std=c++17 -arch=$ARCH -I$CUQ/include"
    export PY=${PY:-$V/bin/python}
}

sg_build() { # make targets, e.g. sg_build single | sg_build single_gate/build/single_bench_gate
    make -sf Makefile "$@" ARCH="-arch=$ARCH"
}

sg_gpu_info() {
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
}

# CUDA-Q is part of the image venv; there is no host/apptainer fallback.
sg_run_cudaq() { # [--env K=V]... <cudaq_probe.py args...>
    local envs=()
    while [ "$1" = "--env" ]; do envs+=("$2"); shift 2; done
    env "${envs[@]}" "$V/bin/python" scripts/baseline/cudaq_probe.py "$@"
}
