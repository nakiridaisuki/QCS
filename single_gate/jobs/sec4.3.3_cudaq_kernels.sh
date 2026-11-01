#!/usr/bin/env bash

# Table 4.3.3 only: which GPU kernels CUDA-Q nvidia runs for qft_n25 fp32,
# fusion on vs off.  nsys truncates kernel names in its table output, so read
# the CSV report and print "kernel_line" rows with the full name.
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
    local rep="single_gate/results/nsys_logs/nsys_kernels_${fusion}_${SLURM_JOB_ID:-local}"
    echo "########## target=nvidia circuit=qft_n25 prec=fp32 fusion=${fusion} ##########"
    "${envargs[@]}" nsys profile -o "$rep" --force-overwrite=true --stats=false \
        "$V/bin/python" scripts/baseline/cudaq_probe.py benchmarks/qft_n25.qasm \
        --target nvidia --warmup 0 --repeats 1 >/dev/null 2>&1
    nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv "$rep.nsys-rep" > "$rep.csv"
    "$V/bin/python" - "$rep.csv" <<'PY'
import csv, sys
lines = open(sys.argv[1], newline="").read().splitlines()
start = next(i for i, l in enumerate(lines) if l.startswith("Time (%)"))
for row in csv.DictReader(lines[start:]):
        name = row.get("Name")
        if not name:
            continue
        inst = int(row.get("Instances") or 0)
        ms = float(row.get("Total Time (ns)") or 0) / 1e6
        print(f"kernel_line : {inst} || {ms:.6f} || {name}")
PY
}

run on
run off
