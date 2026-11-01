#!/usr/bin/env bash

# Table B.0.1 only: swap(23,24) vs swap(0,1), current vs pre-fix legacy binary.
# Deprecated source: single_gate/scripts/legacy/appB/ (commit 18bd1f7).
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

./single_gate/build/single_bench_gate 25 fp32 "${ITERS:-50}" > "$cur"
./single_gate/build/bench_gate_prelane 25 fp32 "${ITERS:-50}" > "$leg"

echo "########## side=current circuit=swap prec=fp32 ##########"
cat "$cur"
echo "########## side=legacy circuit=swap prec=fp32 ##########"
cat "$leg"

"${PY:-$V/bin/python}" - "$cur" "$leg" single_gate/results/tables/appB.0.1_swap_bandwidth.csv <<'PY'
import re, sys
cur, leg, out = sys.argv[1:]

def grab(path, pattern):
    m = re.search(pattern, open(path).read(), re.M)
    return float(m.group(1)) if m else None

high = grab(cur, r"^\s+swap\(\s*23,\s*24\)\s+([0-9.]+) us/gate")
new = grab(cur, r"^\s+local_swap \(perm\)\s+([0-9.]+) us/gate")
old = grab(leg, r"^\s+local_swap \(perm\)\s+([0-9.]+) us/gate")

state_mb = (1 << 25) * 8 / 1e6     # n=25 fp32
rows = [
    ("swap(23,24) current", high, state_mb),
    ("swap(0,1) current", new, 2 * state_mb),
    ("swap(0,1) legacy", old, 2 * state_mb),
]
with open(out, "w") as f:
    f.write("case,us,bytes_mb,effective_gbs\n")
    for name, us, mb in rows:
        if us:
            f.write(f"{name},{us:.1f},{mb:.1f},{mb * 1e3 / us:.0f}\n")
print(open(out).read())
PY
