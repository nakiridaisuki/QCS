# 單閘 kernel 研究（SINGLE_GATE_REPORT）

這個資料夾放 `SINGLE_GATE_REPORT.md` 的所有可重現資料：一支表一支腳本，腳本輸出 log，
再由 parser 轉成 CSV。報告本文仍在 repo 根目錄，這裡只放產生報告數字的工具與結果。

## 目錄結構

```
single_gate/
├── README.md
├── build/                     # make single 的 binary（gitignored）
├── env/                       # 版本清單與各機器路徑
├── jobs/                      # 每張表一支 bash/SLURM script
├── results/
│   ├── raw/                   # <table-id>_<short>-<jobid|local>.out
│   ├── tables/                # <table-id>_<short>.csv、unparsed.txt
│   └── nsys_logs/             # nsys profile 的 .nsys-rep / .sqlite / csv
└── scripts/
    ├── common.sh              # 共用的路徑/環境/bootstrap
    ├── parse_data.py          # 3/4/6/7 章的 CSV 與 -t 輸出
    ├── local_gate.cu          # 現行 local gate simulator
    ├── cuquantum_baseline.cu  # 逐閘 custatevecApplyMatrix 的 baseline
    ├── bench_gate.cu          # 單閘 microbenchmark（ours vs cuStateVec）
    ├── qiskit_qasm_parser.py  # QASM → .circuit IR
    ├── circuits_native/       # native cp/swap IR（gitignored，make 自動生成）
    ├── circuits_ucx/          # u/cx IR（gitignored，make 自動生成）
    └── legacy/
        ├── appB/              # 附錄 B 用：pre-fix 冷凍版本（deprecated）
        └── appC/              # 附錄 C 用：pre-wide 冷凍版本（deprecated）
```

`results/` 被 `.gitignore` 忽略；`raw/`、`tables/`、`nsys_logs/` 都是本地產生的工作資料。
`benchmarks/*.qasm` 與兩個 `circuits_*/` 同樣 gitignored：電路用 `make benchmarks`
從 MQT Bench 下載（需網路），IR 由 make 在跑表前自動從 qasm 生成。

## 命名規則

表號沿用報告的「章節.小節.序號」：

- 正文表：`3.0.1`、`4.1.1`、`7.3.2` …
- 附錄表：`B.0.1`、`C.0.3` …

同一個表號對應三個同名檔案：

```
single_gate/jobs/sec3.0.1_whole_circuit.sh
single_gate/results/raw/sec3.0.1_whole_circuit-<jobid|local>.out
single_gate/results/tables/sec3.0.1_whole_circuit.csv
```

附錄表用 `appB.*` / `appC.*` 前綴（例如 `appC.0.3_wide_shfl`），輸出位置相同，
只是 CSV 由各自的 script 直接寫出，不經過 `parse_data.py`。

## 表與腳本對照

| 表 | 腳本 | 環境 |
|---|---|---|
| 3.0.1 | `sec3.0.1_whole_circuit.sh` | GPU |
| 4.1.1 | `sec4.1.1_cudaq_statevector.sh` | CUDA-Q |
| 4.2.1 | `sec4.2.1_cudaq_tensornet.sh` | CUDA-Q |
| 4.3.1 | `sec4.3.1_cudaq_fusion.sh` | CUDA-Q |
| 4.3.2 | `sec4.3.2_cudaq_kernel_sum.sh` | CUDA-Q + nsys |
| 4.3.3 | `sec4.3.3_cudaq_kernels.sh` | CUDA-Q + nsys |
| 6.2.1 | `sec6.2.1_native_speedup.sh` | GPU |
| 7.0.1 | `sec7.0.1_same_decomposition.sh` | GPU + CUDA-Q |
| 7.1.1 | `sec7.1.1_d2h.sh` | GPU |
| 7.2.1 | `sec7.2.1_kernel_sums.sh` | GPU + CUDA-Q + nsys |
| 7.2.2 | `sec7.2.2_per_operation.sh` | GPU + CUDA-Q + nsys |
| 7.2.3 | `sec7.2.3_per_gate.sh` | GPU |
| 7.3.1 | `sec7.3.1_u_target_sweep.sh` | GPU |
| 7.3.2 | `sec7.3.2_bit_positions.sh` | GPU |
| B.0.1 | `appB.0.1_swap_bandwidth.sh` | GPU |
| B.0.2 | `appB.0.2_swap_before_after.sh` | GPU |
| B.0.3 | `appB.0.3_swap_correctness.sh` | GPU + numpy |
| C.0.1 | `appC.0.1_u_target_before.sh` | GPU |
| C.0.2 | `appC.0.2_ncu_counters.sh` | GPU + ncu（本機） |
| C.0.3 | `appC.0.3_wide_shfl.sh` | GPU |
| C.0.4 | `appC.0.4_end_to_end.sh` | GPU |

## 執行方式

所有 script 都在 `single_gate/env/Dockerfile` 的 image 內執行；主機只提供
GPU driver 與 Docker/Apptainer。建置與執行指令見
[`env/README.md`](env/README.md)。

```bash
make single 7.3.2             # auto-dispatch
make single 4.3.1 SLURM=1     # Slurm wrapper
```

需要指定容器時：

```bash
make single 7.3.2 RUNNER=apptainer
make single 7.3.2 RUNNER=docker IMAGE=single-gate:cudaq0.16
```

## 用 make 跑表

```bash
make single-docker           # 建 Docker image
make single-apptainer        # Docker image -> SIF（中繼 tar 自動刪除）
# 需要指定 cache/tmp 時：
#   make single-apptainer APPTAINER_TMPDIR=... APPTAINER_CACHEDIR=...
make single 7.3.2            # auto: 已在容器內就直接跑，否則 Docker/Apptainer
make single 7.3              # 7.3 小節全部
make single 7                # 第 7 章全部
make single 4.3.1 SLURM=1    # 送 Slurm wrapper，不阻塞
make single_parse ID=7.3     # 只 parse（也會進容器）
```

`X.Y.Z` / `X.Y` / `X` 都支援，附錄用 `B`、`C.0`；`single_parse ID=all` parse 全部。
容器不存在時只會提示建置方式，不會自動 build。Slurm 用
`single_gate/jobs/slurm_container.sh`；Slurm 資源都在 wrapper 宣告，table script
本身沒有 `#SBATCH`。可在 `SBATCH_ARGS` 加 account 或 `-W`。

## 解析 CSV

3/4/6/7 章的表由 `parse_data.py` 從 `results/raw/` 產生：

```bash
python3 single_gate/scripts/parse_data.py
python3 single_gate/scripts/parse_data.py -t list
python3 single_gate/scripts/parse_data.py -t 7.3.2
```

每次執行會把 `single_gate/results/tables/*.csv` 重寫，並更新 `unparsed.txt`；
`unparsed.txt` 正常應該是空的，有內容代表某支 script 的輸出格式變了。
同一張表重跑多次時 `raw/` 會累積多份 log；parse 一律取**檔名排序最後一份**
（last run wins），不做跨 run 平均或取最小。
附錄 B/C 的 CSV 由 `appB.*` / `appC.*` script 直接寫入，不在 `-t list` 裡。

## 資料流與量測定義

- **3.0.1**：CUDA event 包住 gate 迴圈，只算 GPU 執行時間，不含 H2D/D2H、讀檔與正確性 dump。
- **4.1.1 / 4.3.1**：`cudaq_probe.py` 的 `state prep + D2H`（total），warmup 1、mean of 3。
- **4.2.1**：nvidia 用 total；`tensornet` / `tensornet-mps` 因為 CUDA-Q 0.16 不提供 state vector，只能量 enqueue，CSV 會標 `*`。
- **4.3.2**：nsys GPU kernel 總和。暖機在 trace 外跑 1 次，再 profile 3 次單次執行（`--warmup 0 --repeats 1`）取平均。
- **4.3.3**：nsys kernel 名稱與 template 特化（target 數），單次 trace。
- **7.0.1**：ours 是 CUDA event gate 迴圈；CUDA-Q 欄位是 total/materialize。
- **7.2.1**：nsys kernel sum，3 個 profiled run 取平均（CUDA-Q 暖機在 trace 外；ours 的 binary 內含 1 暖機 + 3 計時，所以 kernel 總和除以 4）。
- **7.2.2**：nsys 單次 trace（`--warmup 0 --repeats 1`），每個 kernel 的平均。
- **7.3.1 / 7.3.2**：`bench_gate` 的 `us/gate`，每個位置一個值。

## 已知限制

- **7.0.1 的 CUDA-Q 欄位**：新場次可能比報告高 10–15%，ours 欄則完全一致。
- **4.2.1 的 `*` 欄**和 nvidia 的 total 不是同一個計時邊界，跨欄比較要看註記。
- **4.3.3 的 kernel 特化**：同一份 image 重跑時 kernel 名稱與 instances 穩定，
  但不同 cuStateVec/CUDA-Q 版本可能選到不同特化；報告的表格仍需和實際 CSV 對照。
- **C.0.2** 需要 `ncu` 與 performance counter 權限；Slurm 節點可能不允許。
- 報告本文的多數數字尚未用這次的 CSV 全面更新；README 只描述工具與資料流。

## 維護注意

- 所有 job 共用 `single_gate/scripts/common.sh`（`sg_init`/`sg_env`/`sg_build`）；
  `ARCH` 預設 `sm_75`，H200/H100 要 `ARCH=sm_90`。
- 根目錄 `make single` 會 include `single_gate/Makefile`，從 `single_gate/scripts/*.cu`
  建置 `single_gate/build/single_*`；single gate 的 make 規則都集中在那個檔案。
- 根目錄 `make tables` 現在只建 fusion 表；單閘表請直接用上面的 script。
- `single_gate/scripts/legacy/` 是附錄 B/C 的冷凍舊版，標記 deprecated，不進正常 build。