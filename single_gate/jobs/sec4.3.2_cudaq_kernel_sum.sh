#!/usr/bin/env bash

# Table 4.3.2 only: qft_n25 fp32 GPU kernel time, fusion on vs off (nsys).
# Prints one "kernel_sum_ms : <sum>" line per mode; the D2H/CPU parts are excluded.
# Same scheme as sec7.2.1: warm up once outside the trace, then average three
# profiled runs (--warmup 0 --repeats 1 each).
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env
sg_gpu_info

if [ ! -x "$V/bin/python" ]; then
    echo "!! CUDA-Q venv not found at $V (set CUDAQ_VENV) -- skipping"; exit 0
fi
if ! command -v nsys >/dev/null; then
    echo "!! nsys not found -- skipping"; exit 0
fi

run() { # fusion
    local fusion=$1 envargs=()
    [ "$fusion" = off ] && envargs=(env CUDAQ_FUSION_MAX_QUBITS=-1 CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS=-1)
    echo "########## target=nvidia circuit=qft_n25 prec=fp32 fusion=${fusion} ##########"
    # warm-up run, unprofiled
    "${envargs[@]}" "$V/bin/python" scripts/baseline/cudaq_probe.py benchmarks/qft_n25.qasm \
        --target nvidia --warmup 1 --repeats 1 >/dev/null 2>&1
    local total=0 one rep r
    for r in 1 2 3; do
        rep="single_gate/results/nsys_logs/nsys_kernsum_${fusion}_${r}_${SLURM_JOB_ID:-local}"
        "${envargs[@]}" nsys profile -o "$rep" --force-overwrite=true --stats=false \
            "$V/bin/python" scripts/baseline/cudaq_probe.py benchmarks/qft_n25.qasm \
            --target nvidia --warmup 0 --repeats 1 >/dev/null 2>&1
        one=$(nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv "$rep.nsys-rep" \
            | awk -F, 'NR > 1 && $2 + 0 > 0 { sum += $2 } END { printf "%.6f\n", sum / 1e6 }')
        total=$(awk -v a="$total" -v b="$one" 'BEGIN { printf "%.6f", a + b }')
    done
    awk -v a="$total" 'BEGIN { printf "kernel_sum_ms : %.3f\n", a / 3 }'
}

run on
run off
