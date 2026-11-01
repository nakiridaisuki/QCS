#!/usr/bin/env bash

# Table B.0.2 only: swap(0,1) before/after the fix, by n and precision.
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env
sg_gpu_info

sg_build single_gate/build/single_bench_gate
nvcc $NVCCFLAGS single_gate/scripts/legacy/appB/bench_gate.cu -o single_gate/build/bench_gate_prelane.tmp \
    -L"$CUQUANTUM_LIB" -l:libcustatevec.so.1 -lcudart -Xlinker -rpath -Xlinker "$CUQUANTUM_LIB"
mv -f single_gate/build/bench_gate_prelane.tmp single_gate/build/bench_gate_prelane

cur=$(mktemp); leg=$(mktemp)
trap 'rm -f "$cur" "$leg"' EXIT

for case in 25:fp32 23:fp32 20:fp32 25:fp64 23:fp64; do
    n=${case%%:*}; p=${case##*:}
    echo "########## side=current circuit=swap prec=${p} ##########"
    ./single_gate/build/single_bench_gate "$n" "$p" "${ITERS:-50}" | tee -a "$cur"
    echo "########## side=legacy circuit=swap prec=${p} ##########"
    ./single_gate/build/bench_gate_prelane "$n" "$p" "${ITERS:-50}" | tee -a "$leg"
done

"${PY:-$V/bin/python}" - "$cur" "$leg" single_gate/results/tables/appB.0.2_swap_before_after.csv <<'PY'
import re, sys
cur, leg, out = sys.argv[1:]

def parse(path):
    text = open(path).read()
    out = {}
    pattern = r"n=(\d+)  precision=(fp32|fp64)(.*?)(?=n=\d+  precision=|\Z)"
    for m in re.finditer(pattern, text, re.S):
        us = re.search(r"^\s+local_swap \(perm\)\s+([0-9.]+) us/gate", m.group(3), re.M)
        if us:
            out[(int(m.group(1)), m.group(2))] = float(us.group(1))
    return out

new = parse(cur)
old = parse(leg)
rows = []
for n, prec in ((25, "fp32"), (23, "fp32"), (20, "fp32"), (25, "fp64"), (23, "fp64")):
    o, c = old.get((n, prec)), new.get((n, prec))
    if o and c:
        rows.append((n, prec, o, c, (1 - c / o) * 100))
with open(out, "w") as f:
    f.write("n,precision,legacy_us,current_us,improvement_pct\n")
    for n, prec, o, c, pct in rows:
        f.write(f"{n},{prec},{o:.1f},{c:.1f},{pct:.0f}\n")
print(open(out).read())
PY
