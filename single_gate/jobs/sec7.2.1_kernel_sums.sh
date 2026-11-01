#!/usr/bin/env bash

# Table 7.2.1 only: qft_n20/23/25 fp32+fp64, pure GPU kernel time (nsys kernel sum),
# our native local gate vs CUDA-Q nvidia fusion off.
# Three profiled runs per side; local_gate itself contains 1 warm-up + 3 timed runs,
# CUDA-Q is warmed up outside the trace and profiled with --warmup 0 --repeats 1.
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env
sg_gpu_info

sg_build single_gate/build/single_local_gate

if ! command -v nsys >/dev/null; then
    echo "!! nsys not found -- skipping"; exit 0
fi
if [ ! -x "$V/bin/python" ]; then
    echo "!! CUDA-Q venv not found at $V (set CUDAQ_VENV) -- skipping"; exit 0
fi

sum_kernels() { # nsys-rep divisor
    nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv "$1" \
        | awk -F, -v d="$2" 'NR > 1 && $2 + 0 > 0 { sum += $2 } END { printf "%.6f\n", sum / d / 1e6 }'
}

for q in qft_n20 qft_n23 qft_n25; do
    for prec in fp32 fp64; do
        echo "########## side=ours circuit=${q} prec=${prec} ##########"
        ./single_gate/build/single_local_gate "single_gate/scripts/circuits_native/${q}.circuit" "$prec" >/dev/null
        total=0
        for r in 1 2 3; do
            rep="single_gate/results/nsys_logs/nsys_721_ours_${q}_${prec}_${r}_${SLURM_JOB_ID:-local}"
            nsys profile -o "$rep" --force-overwrite=true --stats=false \
                ./single_gate/build/single_local_gate "single_gate/scripts/circuits_native/${q}.circuit" "$prec" >/dev/null 2>&1
            # 4 circuit executions in the trace: one warm-up + three timed.
            one=$(sum_kernels "$rep.nsys-rep" 4)
            total=$(awk -v a="$total" -v b="$one" 'BEGIN { printf "%.6f", a + b }')
        done
        awk -v a="$total" 'BEGIN { printf "kernel_sum_ms : %.3f\n", a / 3 }'

        opt=(); [ "$prec" = fp64 ] && opt=(--target-option fp64)
        echo "########## target=nvidia circuit=${q} prec=${prec} fusion=off ##########"
        "$V/bin/python" scripts/baseline/cudaq_probe.py "benchmarks/${q}.qasm" \
            --target nvidia "${opt[@]}" --warmup 1 --repeats 1 >/dev/null 2>&1
        total=0
        for r in 1 2 3; do
            rep="single_gate/results/nsys_logs/nsys_721_cudaq_${q}_${prec}_${r}_${SLURM_JOB_ID:-local}"
            env CUDAQ_FUSION_MAX_QUBITS=-1 CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS=-1 \
                nsys profile -o "$rep" --force-overwrite=true --stats=false \
                "$V/bin/python" scripts/baseline/cudaq_probe.py "benchmarks/${q}.qasm" \
                --target nvidia "${opt[@]}" --warmup 0 --repeats 1 >/dev/null 2>&1
            one=$(sum_kernels "$rep.nsys-rep" 1)
            total=$(awk -v a="$total" -v b="$one" 'BEGIN { printf "%.6f", a + b }')
        done
        awk -v a="$total" 'BEGIN { printf "kernel_sum_ms : %.3f\n", a / 3 }'
    done
done
