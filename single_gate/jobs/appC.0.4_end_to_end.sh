#!/usr/bin/env bash

# Table C.0.4 only: end-to-end A/B of the alignas(2*sizeof(T)) change.
# Both binaries come from the current source in the same job; "before" is the
# same file with the alignment sed-ed out.
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env
sg_gpu_info

sed 's/struct alignas(2 \* sizeof(T)) Complex/struct Complex/' \
    single_gate/scripts/local_gate.cu > single_gate/build/local_gate_noalign.cu
nvcc $NVCCFLAGS single_gate/build/local_gate_noalign.cu -o single_gate/build/sim_noalign
nvcc $NVCCFLAGS single_gate/scripts/local_gate.cu -o single_gate/build/sim_wide

raw=$(mktemp); trap 'rm -f "$raw"' EXIT
for rep in 1 2 3; do
    for circ in qft_n20 qft_n23 qft_n25; do
        for prec in fp32 fp64; do
            for side in noalign wide; do
                echo "--- ${circ} ${prec} ${side} rep${rep}" | tee -a "$raw"
                ./single_gate/build/sim_$side "single_gate/scripts/circuits_native/${circ}.circuit" "$prec" \
                    | tee -a "$raw"
            done
        done
    done
done

"${PY:-$V/bin/python}" - "$raw" single_gate/results/tables/appC.0.4_end_to_end.csv <<'PY'
import re, sys
text = open(sys.argv[1]).read()
vals = {}
block = re.compile(r"--- (qft_n\d+) (fp32|fp64) (noalign|wide) rep\d+.*?GPU simulation time : ([0-9.]+) ms", re.S)
for m in block.finditer(text):
    vals.setdefault((m.group(1), m.group(2), m.group(3)), []).append(float(m.group(4)))
with open(sys.argv[2], "w") as f:
    f.write("circuit,fp32_before,fp32_after,fp64_before,fp64_after\n")
    for circ in ("qft_n20", "qft_n23", "qft_n25"):
        row = [circ]
        for prec in ("fp32", "fp64"):
            for side in ("noalign", "wide"):
                xs = vals.get((circ, prec, side), [])
                row.append(f"{sum(xs) / len(xs):.3f}" if xs else "")
        f.write(",".join(row) + "\n")
print(open(sys.argv[2]).read())
PY
