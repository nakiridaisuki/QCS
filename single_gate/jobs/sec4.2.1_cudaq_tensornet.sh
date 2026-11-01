#!/usr/bin/env bash

# Table 4.2.1 only: CUDA-Q nvidia / tensornet / tensornet-mps on vqe_n16 and
# qft_n20.  The number is "get_state enqueue", the only timing all three
# targets have (tensornet has no state vector to materialise).
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env
sg_gpu_info

if [ ! -x "$V/bin/python" ]; then
    echo "!! CUDA-Q venv not found at $V -- run inside single_gate/env/Dockerfile"; exit 1
fi


run() { # target circuit
    local target=$1 circuit=$2
    echo "########## target=${target} circuit=${circuit} prec=fp32 fusion=on ##########"
    sg_run_cudaq "benchmarks/${circuit}.qasm" --target "$target" --warmup 1 --repeats 3 2>&1
}

for q in vqe_n16 qft_n20; do
    run nvidia "$q"
    run tensornet "$q"
    run tensornet-mps "$q"
done
