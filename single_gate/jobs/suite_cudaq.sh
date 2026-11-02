#!/usr/bin/env bash

# CUDA-Q wall-time suite: nvidia fusion on/off on qft_n20/23/25 (fp32+fp64)
# plus tensornet/tensornet-mps on qft_n20.  JIT-dominated; feeds 4.1.1, 4.2.1,
# 4.3.1 and the CUDA-Q column of 7.0.1.  Not needed for the final table.
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env
sg_gpu_info

if [ ! -x "$V/bin/python" ]; then
    echo "!! CUDA-Q venv not found at $V -- run inside single_gate/env/Dockerfile"; exit 1
fi

run_cudaq() { # target circuit prec fusion
    local target=$1 q=$2 prec=$3 fusion=$4
    local envargs=()
    [ "$fusion" = off ] && envargs=(--env CUDAQ_FUSION_MAX_QUBITS=-1 --env CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS=-1)
    local opt=(); [ "$prec" = fp64 ] && opt=(--target-option fp64)
    echo "########## target=${target} circuit=${q} prec=${prec} fusion=${fusion} ##########"
    sg_run_cudaq "${envargs[@]}" \
        "benchmarks/${q}.qasm" --target "$target" "${opt[@]}" --warmup 1 --repeats 3 2>&1
}

for q in qft_n20 qft_n23 qft_n25; do
    for prec in fp32 fp64; do
        run_cudaq nvidia "$q" "$prec" on
        run_cudaq nvidia "$q" "$prec" off
    done
done
# 4.2.1's nvidia column reuses the fusion-on run above; only the tensor-network
# targets are extra.
for t in tensornet tensornet-mps; do
    run_cudaq "$t" qft_n20 fp32 on
done
