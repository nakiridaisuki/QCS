#!/usr/bin/env bash

# Table 4.1.1 only: CUDA-Q nvidia state vector, qft_n20/23/25, fp32/fp64.
# The number is cudaq_probe's "state prep + D2H" (wall time including the copy).
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env
sg_gpu_info

if [ ! -x "$V/bin/python" ]; then
    echo "!! CUDA-Q venv not found at $V -- run inside single_gate/env/Dockerfile"; exit 1
fi


run() { # circuit precision
    local circuit=$1 prec=$2
    local opt=(); [ "$prec" = fp64 ] && opt=(--target-option fp64)
    echo "########## target=nvidia circuit=${circuit} prec=${prec} fusion=on ##########"
    sg_run_cudaq "benchmarks/${circuit}.qasm" --target nvidia "${opt[@]}" --warmup 1 --repeats 3 2>&1
}

for q in qft_n20 qft_n23 qft_n25; do
    for prec in fp32 fp64; do
        run "$q" "$prec"
    done
done
