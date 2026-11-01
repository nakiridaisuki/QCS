# appB: frozen pre-fix swap sources

`18bd1f7` = before `a190b66` (lane kernel) and `f95017b` (aligned Complex): the
"completely uncorrected" scalar swap in table B.0.1/B.0.2.

| file | origin | used by |
|---|---|---|
| `bench_gate.cu` | `18bd1f7:without_gf/src/bench_gate.cu` | `appB.0.1`, `appB.0.2` |
| `test_swap_kernel.py` | `a190b66:scripts/test_swap_kernel.py` | `appB.0.3` |

Do not add these to `make single`; `single_gate/jobs/appB.*.sh` compile
`single_gate/build/bench_gate_prelane` explicitly.
