# Quantum Circuit Simulation — 單閘 kernel 研究

自製 local-gate GPU 模擬器 vs NVIDIA cuStateVec / CUDA-Q 的理想態向量模擬比較。

- 主報告：[SINGLE_GATE_REPORT.md](SINGLE_GATE_REPORT.md)
- 實驗筆記（時間線版，含表號與重現指令）：[single_gate/NOTES.md](single_gate/NOTES.md)
- 重現方式與工具說明：[single_gate/README.md](single_gate/README.md)

```bash
make benchmarks           # 下載電路（MQT Bench，需網路；gitignored）
make single-docker        # 建容器 image（CUDA-Q 0.16 + nvcc + nsys/ncu）
make single final         # 跑最終結果表（自動進容器，缺的量測與 IR 會自動補）
```
