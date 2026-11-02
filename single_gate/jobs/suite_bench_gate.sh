#!/usr/bin/env bash

# bench_gate suite: one full microbenchmark pass per (n, precision).  One run
# feeds tables 7.1.1 (D2H lines), 7.2.3 (per-gate), 7.3.1 (u target sweep) and
# 7.3.2 (bit positions); make only re-runs this when its stamp is missing or
# the inputs changed.
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env
sg_gpu_info

sg_build single_gate/build/single_bench_gate

for n in 20 23 25; do
    for prec in fp32 fp64; do
        ./single_gate/build/single_bench_gate "$n" "$prec" "${ITERS:-50}"
    done
done
