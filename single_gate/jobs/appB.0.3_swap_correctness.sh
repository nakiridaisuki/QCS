#!/usr/bin/env bash

# Table B.0.3 only: swap correctness, numpy full-unitary reference.
# The checker is the frozen copy from f437880^ (single_gate/scripts/legacy).
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env
sg_gpu_info

sg_build single_gate/build/single_local_gate

out=$(mktemp)
trap 'rm -f "$out"' EXIT
"${PY:-$V/bin/python}" single_gate/scripts/legacy/appB/test_swap_kernel.py \
    single_gate/build/single_local_gate | tee "$out"

"${PY:-$V/bin/python}" - "$out" single_gate/results/tables/appB.0.3_swap_correctness.csv <<'PY'
import re, sys
text = open(sys.argv[1]).read()
rows = re.findall(r"^(fp32|fp64): RMSE = (\S+)\s+\(tolerance (\S+)\)", text, re.M)
with open(sys.argv[2], "w") as f:
    f.write("precision,rmse,tolerance,result\n")
    for prec, rmse, tol in rows:
        f.write(f"{prec},{rmse},{tol},{'PASS' if float(rmse) <= float(tol) else 'FAIL'}\n")
print(open(sys.argv[2]).read())
PY
