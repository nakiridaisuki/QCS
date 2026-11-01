#!/usr/bin/env bash

# Table 7.0.1 only: same decomposition, qft_n20/23/25 fp32+fp64.
#   ours     = single_local_gate native cp/swap IR (cudaEvent, gate loop only)
#   cudaq    = nvidia fusion off; materialize and total are printed by the probe
set -euo pipefail

source "$(cd "$(dirname "$0")/../scripts" && pwd)/common.sh"
sg_init
sg_env
sg_gpu_info

sg_build single_gate/build/single_local_gate

if [ ! -x "$V/bin/python" ]; then
    echo "!! CUDA-Q venv not found at $V -- run inside single_gate/env/Dockerfile"; exit 1
fi

# native IR: use the committed file, generate it if missing.  Never fall back
# to u/cx -- that would silently make this table not same-decomposition.
native() {
    local f="single_gate/scripts/circuits_native/$1.circuit"
    if [ ! -f "$f" ]; then
        "${PY:-$V/bin/python}" single_gate/scripts/qiskit_qasm_parser.py --native \
            "benchmarks/$1.qasm" "$f" >/dev/null
    fi
    echo "$f"
}

for q in qft_n20 qft_n23 qft_n25; do
    for prec in fp32 fp64; do
        echo "### ${q} ${prec}"
        echo "-- ours"
        ./single_gate/build/single_local_gate "$(native "$q")" "$prec"

        opt=(); [ "$prec" = fp64 ] && opt=(--target-option fp64)
        echo "########## target=nvidia circuit=${q} prec=${prec} fusion=off ##########"
        sg_run_cudaq --env CUDAQ_FUSION_MAX_QUBITS=-1 --env CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS=-1 \
            "benchmarks/${q}.qasm" --target nvidia "${opt[@]}" --warmup 1 --repeats 3 2>&1
    done
done
