#!/usr/bin/env bash

# Table 6.2.1 only: qft_n20/23/25 fp32+fp64, once through the u/cx IR and
# once through the native cp/swap IR. local_gate itself does 1 warm-up + 3 timed runs.
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env
sg_gpu_info

sg_build single

# u/cx IR is generated; the native cp/swap IR is committed
mkdir -p single_gate/scripts/circuits_ucx
for q in qft_n20 qft_n23 qft_n25; do
    [ -f "single_gate/scripts/circuits_ucx/$q.circuit" ] && continue
    "${PY:-$V/bin/python}" single_gate/scripts/qiskit_qasm_parser.py \
        "benchmarks/$q.qasm" "single_gate/scripts/circuits_ucx/$q.circuit" >/dev/null
done

for q in qft_n20 qft_n23 qft_n25; do
    for prec in fp32 fp64; do
        echo "### ${q} ${prec}"
        for ir in ucx native; do
            echo "-- ${ir}"
            ./single_gate/build/single_local_gate "single_gate/scripts/circuits_${ir}/${q}.circuit" "$prec"
        done
    done
done
