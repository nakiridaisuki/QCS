#!/usr/bin/env bash

# Table C.0.2 only: ncu counters for one u kernel launch, n=23, before the fix
# (frozen a197aaa source).  Counters are clock-independent, so the numbers match
# the H200 table even on a T1000; the cluster has no ncu, run this on a box that
# has it.  skip=2 is u(t=0), skip=4 is u(t=1) (local_u runs twice first).
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env
sg_gpu_info

nvcc $NVCCFLAGS single_gate/scripts/legacy/appC/bench_gate.cu -o single_gate/build/bench_gate_prewide.tmp \
    -L"$CUQUANTUM_LIB" -l:libcustatevec.so.1 -lcudart -Xlinker -rpath -Xlinker "$CUQUANTUM_LIB"
mv -f single_gate/build/bench_gate_prewide.tmp single_gate/build/bench_gate_prewide

METRICS=l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sectors_op_read.sum,dram__bytes_read.sum

tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
: > "$tmp"
for spec in "fp64 2 t=0" "fp64 4 t>=1 (t=1)" "fp32 2 t=0"; do
    set -- $spec
    echo "########## ncu prec=$1 skip=$2 $3 ##########" | tee -a "$tmp"
    ncu --clock-control none --target-processes all --metrics "$METRICS" \
        --launch-skip "$2" --launch-count 1 \
        ./single_gate/build/bench_gate_prewide 23 "$1" 1 >> "$tmp" 2>&1
done
cat "$tmp"

"${PY:-$V/bin/python}" - "$tmp" single_gate/results/tables/appC.0.2_ncu_counters.csv <<'PY'
import re, sys
text = open(sys.argv[1]).read()
# metric values per profile section
sections = re.findall(r"########## ncu prec=(fp32|fp64) skip=\d+ ([^#]+?) ##########(.*?)(?=##########|\Z)", text, re.S)
with open(sys.argv[2], "w") as f:
    f.write("config,l1_ld_requests,l1_ld_sectors,sectors_per_request,l1_st_sectors,l2_read_sectors,dram_read_mb\n")
    for prec, label, body in sections:
        def val(metric):
            m = re.search(rf"^\s*{re.escape(metric)}\s+(.*)$", body, re.M)
            if not m:
                return None
            nums = re.findall(r"[0-9][0-9,]*(?:\.[0-9]+)?", m.group(1))
            return float(nums[-1].replace(",", "")) if nums else None
        req = val("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum")
        ld = val("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
        st = val("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum")
        l2 = val("lts__t_sectors_op_read.sum")
        dram = val("dram__bytes_read.sum")
        if req and ld:
            f.write(f"{prec} {label},{int(req)},{int(ld)},{ld / req:.0f},"
                    f"{int(st) if st else ''},{int(l2) if l2 else ''},{dram if dram else ''}\n")
print(open(sys.argv[2]).read())
PY
