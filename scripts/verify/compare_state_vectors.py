#!/usr/bin/env python3
"""Compare two simulator state-vector dumps (text, or the --out-bin binary form)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

MAGIC = b"QUV1"


def read_state(path: Path) -> np.ndarray:
    """Return the dump as a complex128 vector.

    Both formats are parsed by numpy on purpose: the previous line-by-line Python
    loop took minutes per file at n=25, and the dumps are 1.1 GB of text (268 MB
    in the binary form, which is one write in the simulator).
    """
    with path.open("rb") as stream:
        head = stream.read(16)
    if head[:4] == MAGIC:
        dtype = np.complex64 if int.from_bytes(head[4:8], "little") == 0 else np.complex128
        return np.fromfile(path, dtype=dtype, offset=16).astype(np.complex128)

    data = path.read_bytes()
    i = 0
    while data.startswith(b"#", i):
        i = data.index(b"\n", i) + 1
    flat = np.fromstring(data[i:].decode("ascii"), dtype=np.float64, sep=" ")
    if flat.size % 3:
        raise ValueError(f"{path}: expected triples of index, real, imag")
    rows = flat.reshape(-1, 3)
    index = rows[:, 0].astype(np.int64)
    if not np.array_equal(index, np.arange(index.size)):
        raise ValueError(f"{path}: indices are not 0..n-1 in order")
    return rows[:, 1] + 1j * rows[:, 2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="reference state vector")
    parser.add_argument("candidate", type=Path, help="state vector to check")
    parser.add_argument(
        "--rmse-tolerance",
        type=float,
        required=True,
        help="maximum allowed complex-amplitude RMSE",
    )
    args = parser.parse_args()
    if args.rmse_tolerance < 0.0 or not np.isfinite(args.rmse_tolerance):
        raise ValueError("--rmse-tolerance must be a finite non-negative number")

    baseline = read_state(args.baseline)
    candidate = read_state(args.candidate)
    if baseline.shape != candidate.shape:
        raise ValueError(
            f"different numbers of amplitudes: {baseline.size} != {candidate.size}"
        )

    error = candidate - baseline
    if not np.isfinite(error).all():
        raise ValueError("non-finite amplitude difference")

    rmse = float(np.sqrt(np.mean(np.abs(error) ** 2)))
    passed = rmse <= args.rmse_tolerance
    print(f"Amplitude RMSE      : {rmse:.12g}")
    print(f"RMSE tolerance      : {args.rmse_tolerance:.12g}")
    print(f"Comparison result   : {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)