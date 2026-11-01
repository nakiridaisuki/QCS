# DEPRECATED (appendix B reproduction only)
# Frozen copy of scripts/test_swap_kernel.py as of a190b66, restored because the
# file was deleted with the first-generation pipeline.  Run it against the
# current single_gate/build/single_local_gate from single_gate/jobs/appB.0.3_swap_correctness.sh.
#!/usr/bin/env python3
"""Regression check for the local-gate swap kernel.

Builds a tiny circuit that exercises both swap code paths (intra-chunk lane
permutation and the generic scalar kernel), runs without_gf_simulator on it and
compares the state vector with an independent full-unitary reference computed
with numpy.

Usage: python3 scripts/test_swap_kernel.py [path/to/without_gf_simulator]
"""

from __future__ import annotations

import cmath
import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

N = 6


def u_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
        dtype=complex,
    )


def one_qubit(n: int, q: int, mat: np.ndarray) -> np.ndarray:
    """Full 2^n unitary for `mat` on qubit q (qubit 0 = LSB)."""
    # tensor order is most-significant-qubit first, so qubit q sits at position n-1-q
    left = np.eye(2 ** (n - 1 - q), dtype=complex)
    right = np.eye(2 ** q, dtype=complex)
    return np.kron(np.kron(left, mat), right)


def permutation(n: int, perm) -> np.ndarray:
    dim = 2 ** n
    mat = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        mat[perm(i), i] = 1.0
    return mat


def swap_bits(i: int, a: int, b: int) -> int:
    if ((i >> a) & 1) != ((i >> b) & 1):
        i ^= (1 << a) | (1 << b)
    return i


def cx_bits(i: int, control: int, target: int) -> int:
    if (i >> control) & 1:
        i ^= 1 << target
    return i


def cp_matrix(n: int, control: int, target: int, phase: complex) -> np.ndarray:
    mat = np.eye(2 ** n, dtype=complex)
    for i in range(2 ** n):
        if ((i >> control) & 1) and ((i >> target) & 1):
            mat[i, i] = phase
    return mat


GATES = [
    ("u", 5, u_matrix(1.2, 0.3, -0.4)),
    ("cp", 0, 3, cmath.exp(0.7j)),
    ("cx", 2, 4, None),
    ("swap", 0, 1, None),  # intra-chunk path (fp32)
    ("swap", 0, 5, None),
    ("u", 3, u_matrix(0.6, -1.1, 2.0)),
    ("cx", 0, 5, None),
    ("swap", 4, 5, None),
    ("cp", 1, 4, cmath.exp(-0.3j)),
    ("swap", 1, 2, None),
]


def write_circuit(path: Path) -> None:
    lines = ["QISKIT_CIRCUIT_V1", f"qubits {N}", "global_phase 1 0", f"gates {len(GATES)}"]
    for gate in GATES:
        kind = gate[0]
        if kind == "u":
            _, target, m = gate
            vals = [m[0, 0], m[0, 1], m[1, 0], m[1, 1]]
            flat = " ".join(f"{float(v.real)!r} {float(v.imag)!r}" for v in vals)
            lines.append(f"u {target} {flat}")
        elif kind == "cx":
            lines.append(f"cx {gate[1]} {gate[2]}")
        elif kind == "cp":
            lines.append(f"cp {gate[1]} {gate[2]} {float(gate[3].real)!r} {float(gate[3].imag)!r}")
        elif kind == "swap":
            lines.append(f"swap {gate[1]} {gate[2]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reference() -> np.ndarray:
    psi = np.zeros(2 ** N, dtype=complex)
    psi[0] = 1.0
    for gate in GATES:
        kind = gate[0]
        if kind == "u":
            mat = one_qubit(N, gate[1], gate[2])
        elif kind == "cx":
            mat = permutation(N, lambda i, c=gate[1], t=gate[2]: cx_bits(i, c, t))
        elif kind == "cp":
            mat = cp_matrix(N, gate[1], gate[2], gate[3])
        else:
            mat = permutation(N, lambda i, a=gate[1], b=gate[2]: swap_bits(i, a, b))
        psi = mat @ psi
    return psi


def read_output(path: Path) -> np.ndarray:
    psi = np.zeros(2 ** N, dtype=complex)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        idx, real, imag = line.split()
        psi[int(idx)] = complex(float(real), float(imag))
    return psi


def main() -> int:
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else "./without_gf_simulator").resolve()
    if not binary.exists():
        print(f"ERROR: {binary} not found (run 'make without-gf')")
        return 2

    expected = reference()
    failures = 0
    for precision, tolerance in (("fp32", 1e-5), ("fp64", 1e-12)):
        with tempfile.TemporaryDirectory() as tmp:
            circuit = Path(tmp) / "swap_test.circuit"
            out = Path(tmp) / "swap_test.out"
            write_circuit(circuit)
            subprocess.run([str(binary), str(circuit), precision, str(out)], check=True)
            actual = read_output(out)

        rmse = float(np.sqrt(np.mean(np.abs(expected - actual) ** 2)))
        print(f"{precision}: RMSE = {rmse:.3e}  (tolerance {tolerance:g})")
        if rmse > tolerance:
            worst = int(np.argmax(np.abs(expected - actual)))
            print(f"  worst index {worst}: expected {expected[worst]}, got {actual[worst]}")
            failures += 1

        # units of each permutation-based gate must be preserved exactly
        assert abs(np.linalg.norm(actual) - 1.0) < 1e-5, "norm drifted"

    if failures:
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())