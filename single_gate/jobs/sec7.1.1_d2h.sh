#!/usr/bin/env bash

# Table 7.1.1 only: cudaMemcpy D2H of the whole state vector, n=20/23/25,
# both precisions.  bench_gate prints one "D2H memcpy ... ms (... GB/s, ... MB)".
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env
sg_gpu_info

sg_build single_gate/build/single_bench_gate

for n in 20 23 25; do
    for prec in fp32 fp64; do
        ./single_gate/build/single_bench_gate "$n" "$prec" "${ITERS:-20}"
    done
done
