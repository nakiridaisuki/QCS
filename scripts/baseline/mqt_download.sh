#!/usr/bin/env bash
# Download one MQT Bench circuit (indep level, qiskit compiler) from
# https://www.cda.cit.tum.de/mqtbench/ into benchmarks/<key>_n<n>.qasm
#
#   scripts/baseline/mqt_download.sh qft_n20
#
# The site's files end with barrier/measure lines; those are commented out
# (not deleted) so the result matches the state-vector-only inputs the
# simulators expect.
# Verified byte-identical (modulo those lines) against the previously
# committed files: graph_state_n25, grover_n17 (v-chain), qaoa_n16,
# qft_n20/23/25/30, vqe_n16.
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # repo root

key=${1:?usage: mqt_download.sh <key_nN>, e.g. qft_n20}
name=${key%_n*}
n=${key##*_n}

# key -> selectBench index + display name on the MQT Bench form.
# Verified byte-identical (modulo trailing newline) for: graph_state, grover,
# qaoa, qft, vqe.  The rest use the same form; a rejected request makes the
# zipfile step fail loudly.
case "$name" in
    ae)               idx=1;  disp="Amplitude Estimation (AE)" ;;
    dj)               idx=2;  disp="Deutsch-Jozsa" ;;
    graph_state)      idx=3;  disp="Graph State" ;;
    ghz)              idx=4;  disp="GHZ State" ;;
    grover_noancilla) idx=5;  disp="Grover's (no ancilla)" ;;
    grover)           idx=6;  disp="Grover's (v-chain)" ;;
    portfolio_qaoa)   idx=7;  disp="Portfolio Optimization with QAOA" ;;
    portfolio_vqe)    idx=8;  disp="Portfolio Optimization with VQE" ;;
    qaoa)             idx=9;  disp="Quantum Approximation Optimization Algorithm (QAOA)" ;;
    qft)              idx=10; disp="Quantum Fourier Transformation (QFT)" ;;
    qft_entangled)    idx=11; disp="QFT Entangled" ;;
    qnn)              idx=12; disp="Quantum Neural Network (QNN)" ;;
    qpe_exact)        idx=13; disp="Quantum Phase Estimation (QPE) exact" ;;
    qpe_inexact)      idx=14; disp="Quantum Phase Estimation (QPE) inexact" ;;
    qwalk_noancilla)  idx=15; disp="Quantum Walk (no ancilla)" ;;
    qwalk_vchain)     idx=16; disp="Quantum Walk (v-chain)" ;;
    random)           idx=17; disp="Random Circuit" ;;
    vqe)              idx=18; disp="Variational Quantum Eigensolver (VQE)" ;;
    su2)              idx=19; disp="Efficient SU2 ansatz with Random Parameters" ;;
    realamplitudes)   idx=20; disp="Real Amplitudes ansatz with Random Parameters" ;;
    twolocal)         idx=21; disp="Two Local ansatz with Random Parameters" ;;
    wstate)           idx=22; disp="W-State" ;;
    groundstate)      idx=23; disp="Ground State" ;;
    pricing_call)     idx=24; disp="Pricing Call Option" ;;
    pricing_put)      idx=25;  disp="Pricing Put Option" ;;
    routing)          idx=26; disp="Routing" ;;
    shor)             idx=27; disp="Shor's" ;;
    tsp)              idx=28; disp="Travelling Salesman" ;;
    *) echo "!! unknown benchmark: $name" >&2; exit 2 ;;
esac

out="benchmarks/${key}.qasm"
mkdir -p benchmarks
zip=$(mktemp --suffix=.zip)
trap 'rm -f "$zip"' EXIT

curl -sf -X POST "https://www.cda.cit.tum.de/mqtbench/download" \
    --data-urlencode "selectBench_${idx}=${disp}" \
    -d "minQubits=${n}&maxQubits=${n}&indep_qiskit_compiler=on" \
    -o "$zip"

python3 - "$zip" "$out" <<'PY'
import re, sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
text = z.read(z.namelist()[0]).decode()
out = []
for line in text.splitlines():
    if re.match(r"^(barrier|measure)", line):
        line = "// " + line
    out.append(line)
with open(sys.argv[2], "w") as fh:
    fh.write("\n".join(out) + "\n")
PY
echo "benchmarks: $out"
