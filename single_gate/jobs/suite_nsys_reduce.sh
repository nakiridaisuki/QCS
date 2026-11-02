#!/usr/bin/env bash

# nsys suite, CPU phase: turn the traces left by suite_nsys_trace.sh into
# kernel_sum / op_line / kernel_line rows.  nsys stats needs no GPU, so this
# runs fine on a login node.  Feeds 7.2.1, 7.2.2 (per-op, per-instance
# averages), 4.3.2 and 4.3.3 (kernel families from run 1 of each mode so
# instance counts stay per-execution).
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env

if ! command -v nsys >/dev/null; then
    echo "!! nsys not found -- skipping"; exit 0
fi
if [ ! -x "$V/bin/python" ]; then
    echo "!! CUDA-Q venv not found at $V (set CUDAQ_VENV) -- skipping"; exit 0
fi

NSYS_LOGS=single_gate/results/nsys_logs

need() { # trace-stem
    [ -f "$1.nsys-rep" ] || {
        echo "!! missing trace $1.nsys-rep -- run the trace phase on a GPU first" >&2
        exit 1
    }
}

sum_kernels() { # nsys-rep divisor
    nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv "$1" \
        | awk -F, -v d="$2" 'NR > 1 && $2 + 0 > 0 { sum += $2 } END { printf "%.6f\n", sum / d / 1e6 }'
}

kern_csv() { # nsys-rep -> csv on stdout
    nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv "$1"
}

op_lines() { # nsys-rep -- per-op averages (per instance, so re-executions cancel)
    kern_csv "$1" | "$V/bin/python" -c '
import csv, sys
lines = sys.stdin.read().splitlines()
start = next(i for i, l in enumerate(lines) if l.startswith("Time (%)"))
for row in csv.DictReader(lines[start:]):
    name = row.get("Name")
    inst = int(row.get("Instances") or 0)
    total = float(row.get("Total Time (ns)") or 0)
    if name and inst:
        print(f"op_line : {inst} || {total / inst / 1e3:.3f} || {name}")
'
}

kernel_lines() { # nsys-rep -- full kernel names and instance counts (single execution!)
    kern_csv "$1" | "$V/bin/python" -c '
import csv, sys
lines = sys.stdin.read().splitlines()
start = next(i for i, l in enumerate(lines) if l.startswith("Time (%)"))
for row in csv.DictReader(lines[start:]):
    name = row.get("Name")
    if not name:
        continue
    inst = int(row.get("Instances") or 0)
    ms = float(row.get("Total Time (ns)") or 0) / 1e6
    print(f"kernel_line : {inst} || {ms:.6f} || {name}")
'
}

# ---- ours -------------------------------------------------------------------
for q in qft_n20 qft_n23 qft_n25; do
    for prec in fp32 fp64; do
        echo "########## side=ours circuit=${q} prec=${prec} ##########"
        total=0
        for r in 1 2 3; do
            rep="$NSYS_LOGS/nsys_suite_ours_${q}_${prec}_${r}"
            need "$rep"
            # 4 circuit executions in the trace: one warm-up + three timed.
            one=$(sum_kernels "$rep.nsys-rep" 4)
            total=$(awk -v a="$total" -v b="$one" 'BEGIN { printf "%.6f", a + b }')
            # 7.2.2 per-op rows come from run 1 at n25
            [ "$q" = qft_n25 ] && [ "$r" = 1 ] && op_lines "$rep.nsys-rep"
        done
        awk -v a="$total" 'BEGIN { printf "kernel_sum_ms : %.3f\n", a / 3 }'
    done
done

# ---- CUDA-Q -----------------------------------------------------------------
cudaq_reduce() { # fusion circuits...
    local fusion=$1; shift
    for q in "$@"; do
        for prec in fp32 fp64; do
            [ "$fusion" = on ] && { [ "$q" = qft_n25 ] && [ "$prec" = fp32 ] || continue; }
            echo "########## target=nvidia circuit=${q} prec=${prec} fusion=${fusion} ##########"
            local total=0 one rep
            for r in 1 2 3; do
                rep="$NSYS_LOGS/nsys_suite_cudaq_${fusion}_${q}_${prec}_${r}"
                need "$rep"
                one=$(sum_kernels "$rep.nsys-rep" 1)
                total=$(awk -v a="$total" -v b="$one" 'BEGIN { printf "%.6f", a + b }')
                # 7.2.2: run 1 at n25, both precisions
                [ "$q" = qft_n25 ] && [ "$r" = 1 ] && op_lines "$rep.nsys-rep"
                # 4.3.3: n25 fp32, run 1 (instance counts stay per-execution)
                [ "$q" = qft_n25 ] && [ "$prec" = fp32 ] && [ "$r" = 1 ] && kernel_lines "$rep.nsys-rep"
            done
            awk -v a="$total" 'BEGIN { printf "kernel_sum_ms : %.3f\n", a / 3 }'
        done
    done
}

cudaq_reduce off qft_n20 qft_n23 qft_n25
cudaq_reduce on  qft_n25
