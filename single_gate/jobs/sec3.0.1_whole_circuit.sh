#!/usr/bin/env bash

# Table 3.0.1: NVIDIA cuStateVec (cuquantum_baseline) vs the local-gate simulator
# (without_gf_simulator), one circuit/precision per row.  Runs every row that fits
# in VRAM, smallest first, stops at the first one that does not, and checks the
# state vector against cuStateVec after every row. No options.
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env
sg_gpu_info

# The device peak is the state vector plus a small cuStateVec workspace; the
# host copy the drivers make does not count against VRAM.
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

nvcc --version

sg_build single

BASE=./single_gate/build/single_baseline
LOCAL=./single_gate/build/single_local_gate

timing() { # circuit precision
    local circ=$1 prec=$2
    local ref="single_gate/results/base_${circ}_${prec}.bin"
    local got="single_gate/results/local_${circ}_${prec}.bin"
    local tol=1e-5; [ "$prec" = fp64 ] && tol=1e-12
    echo "### ${circ} ${prec}"
    echo "-- baseline"; $BASE  "single_gate/scripts/circuits_ucx/${circ}.circuit" "$prec" "$ref"
    echo "-- local";    $LOCAL "single_gate/scripts/circuits_ucx/${circ}.circuit" "$prec" "$got"
    if ! "${PY:-$V/bin/python}" scripts/verify/compare_state_vectors.py \
            "$ref" "$got" --rmse-tolerance "$tol"; then
        echo "!! correctness FAILED for ${circ} ${prec}; dumps kept: ${ref} ${got}" >&2
        exit 1
    fi
    rm -f "$ref" "$got"
}

# Exactly the rows of table 3.0.1, smallest first.
CASES="vqe_n16:fp32
vqe_n16:fp64
qft_n20:fp32
qft_n20:fp64
qft_n23:fp32
qft_n23:fp64
qft_n25:fp32
qft_n25:fp64"

echo "### free VRAM $(gpu_free_mib) MiB"
for case in $CASES; do
    circ=${case%%:*}; prec=${case##*:}
    n=${circ##*_n}
    need=$(state_mib "$n" "$prec")
    if ! fits "$n" "$prec"; then
        echo "!! stop at ${circ} ${prec}: needs ${need} MiB + 25% margin, free $(gpu_free_mib) MiB"
        exit 0
    fi
    echo "-> ${circ} ${prec}: state ${need} MiB, free $(gpu_free_mib) MiB"
    timing "$circ" "$prec"
done
