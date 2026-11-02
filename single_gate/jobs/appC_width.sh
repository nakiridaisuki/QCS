#!/usr/bin/env bash

# Appendix C tables C.0.1 + C.0.3 in one pass: build the pre-wide legacy
# bench_gate once (with the extended u target set from C.0.1's sed), run n=25
# fp32+fp64 once, and write both CSVs.  C.0.3's "after" side is bench_prod
# (production kernels).
# Deprecated source: single_gate/scripts/legacy/appC/ (commit a197aaa).
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env
sg_gpu_info

sed 's/{0, 1, 2, n\/2, n-2, n-1}/{0, 1, 2, 3, 4, n\/2, n-2, n-1}/' \
    single_gate/scripts/legacy/appC/bench_gate.cu > single_gate/build/bench_gate_appc_ext.cu
nvcc $NVCCFLAGS single_gate/build/bench_gate_appc_ext.cu -o single_gate/build/bench_gate_prewide.tmp \
    -L"$CUQUANTUM_LIB" -l:libcustatevec.so.1 -lcudart -Xlinker -rpath -Xlinker "$CUQUANTUM_LIB"
mv -f single_gate/build/bench_gate_prewide.tmp single_gate/build/bench_gate_prewide
nvcc $NVCCFLAGS -Isingle_gate/scripts single_gate/scripts/bench_prod.cu -o single_gate/build/bench_prod

before=$(mktemp); after=$(mktemp); trap 'rm -f "$before" "$after"' EXIT
for prec in fp32 fp64; do
    echo "########## side=before prec=${prec} ##########"
    ./single_gate/build/bench_gate_prewide 25 "$prec" "${ITERS:-50}" | tee -a "$before"
done
echo "########## side=after prec=fp64 ##########"
./single_gate/build/bench_prod 25 fp64 "${ITERS:-50}" | tee -a "$after"

"${PY:-$V/bin/python}" - "$before" "$after" \
    single_gate/results/tables/appC.0.1_u_target_before.csv \
    single_gate/results/tables/appC.0.3_wide_shfl.csv <<'PY'
import re, sys
before, after, out1, out3 = sys.argv[1:]
text_b, text_a = open(before).read(), open(after).read()

def block(text, prec):
    m = re.search(rf"n=25\s+precision={prec}(.*?)(?=n=25\s+precision=|\Z)", text, re.S)
    return m.group(1) if m else ""

def grab(text, label):
    m = re.search(rf"^\s+{re.escape(label)}\s+([0-9.]+) us/gate", text, re.M)
    return float(m.group(1)) if m else None

# C.0.1: u target sweep before the fix
with open(out1, "w") as f:
    f.write("target,fp32_ours,fp64_ours,fp32_custatevec,fp64_custatevec\n")
    for t in (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 23, 24):
        row = [grab(block(text_b, p), f"{k}(t={t:2d})")
               for p in ("fp32", "fp64") for k in ("u", "csv_u")]
        f.write(f"{t}," + ",".join("" if v is None else f"{v}" for v in row) + "\n")
print(open(out1).read())

# C.0.3: before/after at n25 fp64
rows = [
    ("u t=0", "u(t= 0)"), ("u t=1", "u(t= 1)"), ("u t=2", "u(t= 2)"),
    ("swap(0,2)", "swap( 0, 2)"), ("swap(0,1)", "swap( 0, 1)"),
    ("cx(0,2)", "cx( 0, 2)"), ("cx(0,1)", "cx( 0, 1)"),
    ("cp(0,12)", "cp( 0,12)"), ("cp(0,2)", "cp( 0, 2)"),
]
b64, a = block(text_b, "fp64"), text_a
import csv as csvlib  # gate names contain commas
with open(out3, "w", newline="") as f:
    w = csvlib.writer(f)
    w.writerow(["gate", "before_us", "after_us", "improvement_pct"])
    for name, label in rows:
        b, a_ = grab(b64, label), grab(a, label)
        if b and a_:
            w.writerow([name, f"{b:.1f}", f"{a_:.1f}", f"{(1 - a_ / b) * 100:.1f}"])
print(open(out3).read())
PY
