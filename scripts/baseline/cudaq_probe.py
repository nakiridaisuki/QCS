#!/usr/bin/env python3
"""Run an MQT-benchmark OpenQASM 2 circuit end to end on CUDA-Q.

Unlike the project's own simulators (which consume the normalized u/cx IR),
this feeds the *whole* .qasm file to CUDA-Q's compiler and runtime, so the
result is a "full stack CUDA-Q" baseline.  Supported targets include nvidia,
nvidia-fp64, tensornet, tensornet-mps and qpp-cpu.

Optionally dumps the resulting state vector as "index real imag" so the
existing scripts/compare_state_vectors.py can check it against a reference.
"""

from __future__ import annotations

import argparse
import math
import re
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def parse_qasm(path: Path) -> tuple[int, list[tuple[str, list[float], list[int]]]]:
    # A QASM file may declare several registers (grover_n17 has qreg q[16] and
    # qreg flag[1]).  Flatten them into one index space: register name -> base
    # offset.  Taking only one register (the old behaviour) silently produced a
    # 1-qubit vector and CUDA-Q then failed with
    #   'quake.extract_ref' op invalid index [1] because >= size [1]
    num_qubits = 0
    regs: dict[str, int] = {}
    gates: list[tuple[str, list[float], list[int]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        match = re.match(r"qreg\s+(\w+)\[(\d+)\]", line)
        if match:
            regs[match.group(1)] = num_qubits
            num_qubits += int(match.group(2))
            continue
        if re.match(r"(OPENQASM|creg|include|measure|barrier|opaque|if)", line):
            continue
        match = re.match(r"([A-Za-z_]\w*)(?:\(([^)]*)\))?\s+(.+);", line)
        if not match:
            continue
        name, params, args = match.group(1), match.group(2), match.group(3)
        qubits = []
        for reg_name, idx in re.findall(r"([A-Za-z_]\w*)\[(\d+)\]", args):
            if reg_name not in regs:
                raise SystemExit(f"{path}: unknown register {reg_name!r}")
            qubits.append(regs[reg_name] + int(idx))
        values = (
            [eval(p, {"__builtins__": {}}, {"pi": math.pi}) for p in params.split(",")]
            if params
            else []
        )
        gates.append((name, values, qubits))
    if not num_qubits:
        raise SystemExit(f"{path}: no qreg found")
    return num_qubits, gates


def gate_source(name: str, params: list[float], qubits: list[int]) -> str:
    q = [f"q[{i}]" for i in qubits]
    simple = {"h", "x", "y", "z", "s", "t", "sdg", "tdg"}
    if name in simple:
        return f"{name}({q[0]})"
    two = {"cx", "cz", "swap"}
    if name in two:
        return f"{name}({q[0]}, {q[1]})"
    one_param = {"rx", "ry", "rz", "r1"}
    if name in one_param:
        return f"{name}({params[0]!r}, {q[0]})"
    if name == "u":
        # Qiskit u(theta,phi,lam) = Rz(phi) Ry(theta) Rz(lam); CUDA-Q has no u3,
        # so emit the three rotations.  Exact up to a global phase, which does not
        # affect block counting.
        th, ph, la = params
        return f"rz({la!r}, {q[0]}); ry({th!r}, {q[0]}); rz({ph!r}, {q[0]})"
    # qiskit cp(lambda) == controlled-r1(lambda) == diag(1,1,1,e^{i*lambda})
    if name == "cp":
        return f"cr1({params[0]!r}, {q[0]}, {q[1]})"
    raise SystemExit(f"unsupported gate: {name}")


def build_kernel(num_qubits: int, gates, source_path: Path):
    import cudaq  # noqa: F401
    import importlib.util
    import tempfile

    lines = [
        "import cudaq",
        "from cudaq.qis import *",
        "@cudaq.kernel",
        "def circuit():",
        f"    q = cudaq.qvector({num_qubits})",
    ]
    lines += [f"    {gate_source(n, p, q)}" for n, p, q in gates]

    # CUDA-Q's kernel decorator re-reads the function source via inspect, so the
    # generated code must live in a real file (the .qasm is not Python).
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="cudaq_kernel_", delete=False
    ) as handle:
        handle.write("\n".join(lines) + "\n")
        generated = handle.name
    spec = importlib.util.spec_from_file_location("cudaq_generated_circuit", generated)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.circuit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qasm", type=Path)
    parser.add_argument("--target", default="nvidia")
    parser.add_argument("--target-option", default=None, help="e.g. fp64")
    parser.add_argument("--dump", type=Path, help="write state vector here")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    import cudaq
    import numpy as np

    num_qubits, gates = parse_qasm(args.qasm)
    kernel = build_kernel(num_qubits, gates, args.qasm)
    if args.target_option:
        cudaq.set_target(args.target, option=args.target_option)
    else:
        cudaq.set_target(args.target)

    print(f"CUDA-Q              : {cudaq.__version__.split('(')[0].strip()}")
    print(f"Target              : {args.target}" + (f" (option={args.target_option})" if args.target_option else ""))
    print(f"Qubits              : {num_qubits}")
    print(f"Gates               : {len(gates)}")

    # np.asarray only accepts vector/matrix states; tensornet hands back a State
    # whose to_numpy() performs the tensor contraction.  Fall back to that so the
    # total (enqueue + materialise) is measurable for every target.
    def to_array(state):
        try:
            return np.asarray(state), True
        except (RuntimeError, TypeError):
            try:
                return state.to_numpy(), True
            except Exception:
                return None, False

    for _ in range(args.warmup):  # JIT + first executions, excluded from timing
        to_array(cudaq.get_state(kernel))

    sum_enqueue = sum_materialize = sum_total = 0.0
    values = None
    for _ in range(args.repeats):
        start = time.perf_counter()
        state = cudaq.get_state(kernel)
        mid = time.perf_counter()
        got, ok = to_array(state)
        end = time.perf_counter()
        if ok:
            values = got
        sum_enqueue += (mid - start) * 1e3
        sum_materialize += (end - mid) * 1e3
        sum_total += (end - start) * 1e3
    print(f"get_state enqueue   : {sum_enqueue / args.repeats:.3f} ms  (mean of {args.repeats})")
    if values is None:
        print(f"materialize + D2H   : n/a  (target exposes no state vector)")
        print(f"state prep + D2H    : n/a")
    else:
        print(f"materialize + D2H   : {sum_materialize / args.repeats:.3f} ms  (mean of {args.repeats})")
        print(f"state prep + D2H    : {sum_total / args.repeats:.3f} ms  (mean of {args.repeats})")

    if args.dump and values is not None:
        with args.dump.open("w", encoding="utf-8") as stream:
            stream.write("# index real imag\n")
            for index, value in enumerate(values):
                stream.write(f"{index} {value.real:.17g} {value.imag:.17g}\n")
        print(f"State vector dumped : {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())