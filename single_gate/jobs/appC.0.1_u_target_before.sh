#!/usr/bin/env bash

# Table C.0.1 only: u target-position sweep BEFORE the wide/shfl fix
# (frozen a197aaa source, single_gate/scripts/legacy/appC/).
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

raw=$(mktemp); trap 'rm -f "$raw"' EXIT
for prec in fp32 fp64; do
    ./single_gate/build/bench_gate_prewide 25 "$prec" "${ITERS:-50}" >> "$raw"
done
cat "$raw"

"${PY:-$V/bin/python}" - "$raw" single_gate/results/tables/appC.0.1_u_target_before.csv <<'PY'
import re, sys
text = open(sys.argv[1]).read()

def values(text, prec):
    out = {}
    m = re.search(rf"n=25\s+precision={prec}(.*?)(?=n=25\s+precision=|\Z)", text, re.S)
    if not m:
        return out
    for mm in re.finditer(r"^\s+(u|csv_u)\(t=\s*(\d+)\)\s+([0-9.]+) us/gate", m.group(1), re.M):
        out[(mm.group(1), int(mm.group(2)))] = float(mm.group(3))
    return out

f32, f64 = values(text, "fp32"), values(text, "fp64")
with open(sys.argv[2], "w") as f:
    f.write("target,fp32_ours,fp64_ours,fp32_custatevec,fp64_custatevec\n")
    for t in (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 23, 24):
        f.write(f"{t},{f32.get(('u', t), '')},{f64.get(('u', t), '')},"
                f"{f32.get(('csv_u', t), '')},{f64.get(('csv_u', t), '')}\n")
print(open(sys.argv[2]).read())
PY
