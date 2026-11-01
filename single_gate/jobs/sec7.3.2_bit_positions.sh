#!/usr/bin/env bash

# Table 7.3.2 only: swap / cp / cx at every bit position, H200 n=25 fp32,
# plus the matching cuStateVec entries.
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env
sg_gpu_info

sg_build single_gate/build/single_bench_gate

./single_gate/build/single_bench_gate 25 fp32 "${ITERS:-50}"
