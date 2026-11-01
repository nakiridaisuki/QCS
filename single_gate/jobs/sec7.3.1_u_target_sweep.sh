#!/usr/bin/env bash

# Table 7.3.1 only: u at every target position, n=25 fp32/fp64, ours vs cuStateVec.
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env
sg_gpu_info

sg_build single_gate/build/single_bench_gate

for prec in fp32 fp64; do
    ./single_gate/build/single_bench_gate 25 "$prec" "${ITERS:-50}"
done
