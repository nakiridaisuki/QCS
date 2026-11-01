# appC: frozen pre-wide / pre-shfl sources

`a197aaa` = after the swap lane kernel and after the u/bit-position sweeps, but
before `f95017b` (aligned Complex + warp-transpose u).  This is the "before"
side of tables C.0.1/C.0.2/C.0.3; the live `single_gate/scripts/` sources are
"after".

| file | origin | used by |
|---|---|---|
| `bench_gate.cu` | `a197aaa:without_gf/src/bench_gate.cu` | `appC.0.1`, `appC.0.2`, `appC.0.3` |

Do not add these to `make single`; `single_gate/jobs/appC.*.sh` compile
`single_gate/build/bench_gate_prewide` explicitly.
