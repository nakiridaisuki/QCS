# Appendix B/C: frozen pre-fix sources

Deprecated reproduction material for `SINGLE_GATE_REPORT.md` appendices B and C.
Nothing here is built or imported by the normal pipeline.

| directory | commit | used by |
|---|---|---|
| `appB/` | `18bd1f7` | `appB.0.1`, `appB.0.2` (`bench_gate.cu`) |
| `appB/` | `a190b66` | `appB.0.3` (`test_swap_kernel.py`) |
| `appC/` | `a197aaa` | `appC.0.1`, `appC.0.2`, `appC.0.3` |

Each directory has its own README with the exact "before/after" meaning.  The
scripts build them into `single_gate/build/bench_gate_prelane` /
`single_gate/build/bench_gate_prewide` and never touch `make single`.

Outputs keep the normal layout, with the appendix prefix:

    single_gate/results/raw/appB.0.x_<short>-<jobid|local>.out
    single_gate/results/tables/appB.0.x_<short>.csv
    single_gate/results/raw/appC.0.x_<short>-<jobid|local>.out
    single_gate/results/tables/appC.0.x_<short>.csv
