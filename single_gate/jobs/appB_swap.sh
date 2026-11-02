#!/usr/bin/env bash

# Appendix B tables B.0.1 + B.0.2 in one pass: the current and pre-fix legacy
# bench_gate are built once, five cases are run once, and both CSVs come from
# the same outputs (B.0.1 is the n25 fp32 slice; B.0.2 is the five-case
# before/after).  Deprecated source: single_gate/scripts/legacy/appB/.
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

"${PY:-$V/bin/python}" - "$cur" "$leg" \
    single_gate/results/tables/appB.0.1_swap_bandwidth.csv \
    single_gate/results/tables/appB.0.2_swap_before_after.csv <<'PY'
import re, sys
cur, leg, out1, out2 = sys.argv[1:]

def blocks(path):
    text = open(path).read()
    out = {}
    for m in re.finditer(r"n=(\d+)  precision=(fp32|fp64)(.*?)(?=n=\d+  precision=|\Z)",
                         text, re.S):
        out[(int(m.group(1)), m.group(2))] = m.group(3)
    return out

def grab(text, pattern):
    m = re.search(pattern, text, re.M)
    return float(m.group(1)) if m else None

SWAP = r"^\s+local_swap \(perm\)\s+([0-9.]+) us/gate"
c25, l25 = blocks(cur).get((25, "fp32"), ""), blocks(leg).get((25, "fp32"), "")

# B.0.1: n25 fp32 bandwidth framing (csv.writer: the case names contain commas)
import csv as csvlib
state_mb = (1 << 25) * 8 / 1e6
with open(out1, "w", newline="") as f:
    w = csvlib.writer(f)
    w.writerow(["case", "us", "bytes_mb", "effective_gbs"])
    for name, us, mb in (("swap(23,24) current", grab(c25, r"^\s+swap\(\s*23,\s*24\)\s+([0-9.]+) us/gate"), state_mb),
                         ("swap(0,1) current", grab(c25, SWAP), 2 * state_mb),
                         ("swap(0,1) legacy", grab(l25, SWAP), 2 * state_mb)):
        if us:
            w.writerow([name, f"{us:.1f}", f"{mb:.1f}", f"{mb * 1e3 / us:.0f}"])
print(open(out1).read())

# B.0.2: five-case before/after
new, old = blocks(cur), blocks(leg)
with open(out2, "w") as f:
    f.write("n,precision,legacy_us,current_us,improvement_pct\n")
    for n, prec in ((25, "fp32"), (23, "fp32"), (20, "fp32"), (25, "fp64"), (23, "fp64")):
        o = grab(old.get((n, prec), ""), SWAP)
        c = grab(new.get((n, prec), ""), SWAP)
        if o and c:
            f.write(f"{n},{prec},{o:.1f},{c:.1f},{(1 - c / o) * 100:.0f}\n")
print(open(out2).read())
PY
