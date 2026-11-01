# Quantum Circuit Simulation — 單閘 kernel 研究

自製 local-gate GPU 模擬器 vs NVIDIA cuStateVec / CUDA-Q 的理想態向量模擬比較。

- 主報告：[SINGLE_GATE_REPORT.md](SINGLE_GATE_REPORT.md)
- 重現方式與工具說明：[single_gate/README.md](single_gate/README.md)

```bash
make single-docker        # 建容器 image（CUDA-Q 0.16 + nvcc + nsys/ncu）
make single 7.2.1         # 跑其中一張表（自動進容器）
```
