#!/usr/bin/env bash

# Table 7.2.2 only: qft_n25 fp32/fp64, per-operation GPU time (nsys average per
# kernel instance), our native local gate vs CUDA-Q nvidia fusion off.
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

op_lines() { # nsys-rep
    nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv "$1" > "$1.csv"
    "$V/bin/python" - "$1.csv" <<'PY'
import csv, sys
lines = open(sys.argv[1], newline="").read().splitlines()
start = next(i for i, l in enumerate(lines) if l.startswith("Time (%)"))
for row in csv.DictReader(lines[start:]):
    name = row.get("Name")
    inst = int(row.get("Instances") or 0)
    total = float(row.get("Total Time (ns)") or 0)
    if name and inst:
        print(f"op_line : {inst} || {total / inst / 1e3:.3f} || {name}")
PY
}

for prec in fp32 fp64; do
    echo "########## side=ours circuit=qft_n25 prec=${prec} ##########"
    rep="single_gate/results/nsys_logs/nsys_722_ours_${prec}_${SLURM_JOB_ID:-local}"
    nsys profile -o "$rep" --force-overwrite=true --stats=false \
        ./single_gate/build/single_local_gate single_gate/scripts/circuits_native/qft_n25.circuit "$prec" >/dev/null 2>&1
    op_lines "$rep.nsys-rep"

    opt=(); [ "$prec" = fp64 ] && opt=(--target-option fp64)
    echo "########## target=nvidia circuit=qft_n25 prec=${prec} fusion=off ##########"
    rep="single_gate/results/nsys_logs/nsys_722_cudaq_${prec}_${SLURM_JOB_ID:-local}"
    env CUDAQ_FUSION_MAX_QUBITS=-1 CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS=-1 \
        nsys profile -o "$rep" --force-overwrite=true --stats=false \
        "$V/bin/python" scripts/baseline/cudaq_probe.py benchmarks/qft_n25.qasm \
        --target nvidia "${opt[@]}" --warmup 0 --repeats 1 >/dev/null 2>&1
    op_lines "$rep.nsys-rep"
done
