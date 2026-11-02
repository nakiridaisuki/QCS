#!/usr/bin/env bash

# Ours suite: cuStateVec baseline + local gate on the u/cx IR, with the
# per-circuit RMSE check.  Seconds-scale; feeds 3.0.1 and the u/cx column of
# 6.2.1 (and table 0.1/final).  make re-runs this only when its stamp is
# missing or the inputs changed.
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env
sg_gpu_info

sg_build single

BASE=./single_gate/build/single_baseline
LOCAL=./single_gate/build/single_local_gate

# The device peak is the state vector plus a small cuStateVec workspace.
gpu_free_mib() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '
}
state_mib() { # qubits precision
    local width=8; [ "$2" = fp64 ] && width=16
    echo $(( (1 << $1) * width / 1024 / 1024 ))
}
fits() { # qubits precision -- 25% headroom on top of the state, min 256 MiB
    local need; need=$(state_mib "$1" "$2")
    [ "$(gpu_free_mib)" -ge $(( need + need / 4 + 256 )) ]
}

for q in qft_n20 qft_n23 qft_n25; do
    n=${q##*_n}
    for prec in fp32 fp64; do
        echo "### ${q} ${prec}"
        if ! fits "$n" "$prec"; then
            echo "!! skip ${q} ${prec}: needs $(state_mib "$n" "$prec") MiB + margin, free $(gpu_free_mib) MiB"
            continue
        fi
        tol=1e-5; [ "$prec" = fp64 ] && tol=1e-12
        ref="single_gate/results/base_${q}_${prec}.bin"
        got="single_gate/results/local_${q}_${prec}.bin"
        echo "-- baseline"; $BASE  "single_gate/scripts/circuits_ucx/${q}.circuit" "$prec" "$ref"
        echo "-- local";    $LOCAL "single_gate/scripts/circuits_ucx/${q}.circuit" "$prec" "$got"
        if ! "${PY:-$V/bin/python}" scripts/verify/compare_state_vectors.py \
                "$ref" "$got" --rmse-tolerance "$tol"; then
            echo "!! correctness FAILED for ${q} ${prec}; dumps kept: ${ref} ${got}" >&2
            exit 1
        fi
        rm -f "$ref" "$got"
    done
done
