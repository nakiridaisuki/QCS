#!/usr/bin/env bash

# Table 7.2.3 only: one gate of each kind, ours vs the cuStateVec entry points,
# H200 n=25 fp32.  The table is a single bench_gate run.
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env
sg_gpu_info

sg_build single_gate/build/single_bench_gate

./single_gate/build/single_bench_gate 25 fp32 "${ITERS:-50}"
