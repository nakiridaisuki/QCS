#!/usr/bin/env bash

# Table C.0.3 only: before/after the wide + warp-transpose fix, H200 n=25 fp64.
# before = a197aaa source (single_gate/scripts/legacy/appC), after = current production kernels (bench_prod).
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env
sg_gpu_info

nvcc $NVCCFLAGS single_gate/scripts/legacy/appC/bench_gate.cu -o single_gate/build/bench_gate_prewide.tmp \
    -L"$CUQUANTUM_LIB" -l:libcustatevec.so.1 -lcudart -Xlinker -rpath -Xlinker "$CUQUANTUM_LIB"
mv -f single_gate/build/bench_gate_prewide.tmp single_gate/build/bench_gate_prewide
nvcc $NVCCFLAGS -Isingle_gate/scripts single_gate/scripts/bench_prod.cu -o single_gate/build/bench_prod

before=$(mktemp); after=$(mktemp); trap 'rm -f "$before" "$after"' EXIT
./single_gate/build/bench_gate_prewide 25 fp64 "${ITERS:-50}" > "$before"
./single_gate/build/bench_prod 25 fp64 "${ITERS:-50}" > "$after"
echo "########## side=before ##########"; cat "$before"
echo "########## side=after ##########"; cat "$after"

"${PY:-$V/bin/python}" - "$before" "$after" single_gate/results/tables/appC.0.3_wide_shfl.csv <<'PY'
import re, sys

def grab(path, label):
    text = open(path).read()
    m = re.search(rf"^\s+{re.escape(label)}\s+([0-9.]+) us/gate", text, re.M)
    return float(m.group(1)) if m else None

rows = [
    ("u t=0", "u(t= 0)"), ("u t=1", "u(t= 1)"), ("u t=2", "u(t= 2)"),
    ("swap(0,2)", "swap( 0, 2)"), ("swap(0,1)", "swap( 0, 1)"),
    ("cx(0,2)", "cx( 0, 2)"), ("cx(0,1)", "cx( 0, 1)"),
    ("cp(0,12)", "cp( 0,12)"), ("cp(0,2)", "cp( 0, 2)"),
]
with open(sys.argv[3], "w") as f:
    f.write("gate,before_us,after_us,improvement_pct\n")
    for name, label in rows:
        b, a = grab(sys.argv[1], label), grab(sys.argv[2], label)
        if b and a:
            f.write(f"{name},{b:.1f},{a:.1f},{(1 - a / b) * 100:.1f}\n")
print(open(sys.argv[3]).read())
PY
