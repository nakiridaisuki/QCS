#!/usr/bin/env python3
"""Transpile OpenQASM 2 into the compact IR consumed by the GPU simulators."""

from __future__ import annotations

import argparse
import cmath
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse and transpile OpenQASM 2 for the GPU simulators."
    )
    parser.add_argument("qasm", type=Path, help="input OpenQASM 2 file")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="output circuit IR (default: circuits/<qasm-stem>.circuit)",
    )
    parser.add_argument(
        "--native",
        action="store_true",
        help="keep cp/swap as native gates instead of decomposing to u/cx",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.qasm.is_file():
        raise FileNotFoundError(f"QASM file not found: {args.qasm}")

    output = args.output or Path("circuits") / f"{args.qasm.stem}.circuit"
    circuit = QuantumCircuit.from_qasm_file(str(args.qasm))

    non_unitary = {
        instruction.operation.name
        for instruction in circuit.data
        if instruction.operation.name in {"measure", "reset", "if_else", "while_loop", "for_loop"}
    }
    if non_unitary:
        names = ", ".join(sorted(non_unitary))
        raise ValueError(f"state-vector circuit contains non-unitary operations: {names}")

    basis = ["u", "cx", "cp", "swap"] if args.native else ["u", "cx"]
    normalized = transpile(circuit, basis_gates=basis, optimization_level=0)
    operations: list[str] = []
    for instruction in normalized.data:
        operation = instruction.operation
        if operation.name == "barrier":
            continue
        qubits = [normalized.find_bit(qubit).index for qubit in instruction.qubits]
        if operation.name == "cx" and len(qubits) == 2:
            operations.append(f"cx {qubits[0]} {qubits[1]}")
        elif operation.name == "cp" and len(qubits) == 2:
            phase = cmath.exp(1j * float(operation.params[0]))
            operations.append(
                f"cp {qubits[0]} {qubits[1]} {phase.real:.17g} {phase.imag:.17g}"
            )
        elif operation.name == "swap" and len(qubits) == 2:
            operations.append(f"swap {qubits[0]} {qubits[1]}")
        elif len(qubits) == 1:
            matrix = Operator(operation).data.reshape(-1)
            values = " ".join(
                f"{value.real:.17g} {value.imag:.17g}" for value in matrix
            )
            operations.append(f"u {qubits[0]} {values}")
        else:
            raise ValueError(
                f"transpilation left unsupported operation {operation.name!r} "
                f"on {len(qubits)} qubits"
            )

    phase = cmath.exp(1j * float(normalized.global_phase))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        stream.write("QISKIT_CIRCUIT_V1\n")
        stream.write(f"qubits {normalized.num_qubits}\n")
        stream.write(f"global_phase {phase.real:.17g} {phase.imag:.17g}\n")
        stream.write(f"gates {len(operations)}\n")
        stream.write("\n".join(operations))
        stream.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
