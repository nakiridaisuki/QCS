#!/usr/bin/env bash

# nsys suite, GPU phase: produce the traces only.  Analysis lives in
# suite_nsys_reduce.sh (CPU-only, runs fine on a login node).  The unprofiled
# warm-up run of local_gate prints the cudaEvent timing used by the native
# columns of 6.2.1 / 7.0.1.
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

LOCAL=./single_gate/build/single_local_gate
NSYS_LOGS=single_gate/results/nsys_logs

# ---- ours: local_gate, native IR -------------------------------------------
for q in qft_n20 qft_n23 qft_n25; do
    for prec in fp32 fp64; do
        echo "### ${q} ${prec}"
        echo "-- native"
        $LOCAL "single_gate/scripts/circuits_native/${q}.circuit" "$prec"
        for r in 1 2 3; do
            rep="$NSYS_LOGS/nsys_suite_ours_${q}_${prec}_${r}"
            nsys profile -o "$rep" --force-overwrite=true --stats=false \
                $LOCAL "single_gate/scripts/circuits_native/${q}.circuit" "$prec" >/dev/null 2>&1
        done
    done
done

# ---- CUDA-Q -----------------------------------------------------------------
cudaq_side() { # fusion circuits...
    local fusion=$1; shift
    local envargs=()
    [ "$fusion" = off ] && envargs=(env CUDAQ_FUSION_MAX_QUBITS=-1 CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS=-1)
    for q in "$@"; do
        for prec in fp32 fp64; do
            # only n25 fp32 is needed for fusion on (4.3.2/4.3.3)
            [ "$fusion" = on ] && { [ "$q" = qft_n25 ] && [ "$prec" = fp32 ] || continue; }
            local opt=(); [ "$prec" = fp64 ] && opt=(--target-option fp64)
            # warm-up run, unprofiled
            "${envargs[@]}" "$V/bin/python" scripts/baseline/cudaq_probe.py \
                "benchmarks/${q}.qasm" --target nvidia "${opt[@]}" \
                --warmup 1 --repeats 1 >/dev/null 2>&1
            for r in 1 2 3; do
                local rep="$NSYS_LOGS/nsys_suite_cudaq_${fusion}_${q}_${prec}_${r}"
                "${envargs[@]}" nsys profile -o "$rep" --force-overwrite=true --stats=false \
                    "$V/bin/python" scripts/baseline/cudaq_probe.py \
                    "benchmarks/${q}.qasm" --target nvidia "${opt[@]}" \
                    --warmup 0 --repeats 1 >/dev/null 2>&1
            done
        done
    done
}

cudaq_side off qft_n20 qft_n23 qft_n25
cudaq_side on  qft_n25
