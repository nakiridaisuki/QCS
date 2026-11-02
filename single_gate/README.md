# 單閘 kernel 研究（SINGLE_GATE_REPORT）

這個資料夾放 `SINGLE_GATE_REPORT.md` 的所有可重現資料：四支量測套組（suite）負責
產生原始數據，parser 再把 raw log 轉成各張表。報告本文在 repo 根目錄。

## 目錄結構

```
single_gate/
├── README.md
├── build/                     # make single 的 binary（gitignored）
├── env/                       # 容器 image 定義與版本清單
├── jobs/                      # 四支套組腳本 + 附錄表腳本 + SLURM wrapper
├── results/                   # 全部 gitignored
│   ├── raw/                   # <腳本名>-<jobid|local>.out、stamp 檔
│   ├── tables/                # <表號>.csv、unparsed.txt
│   └── nsys_logs/             # nsys 的 .nsys-rep / .sqlite
└── scripts/
    ├── common.sh              # 共用的路徑/環境/bootstrap
    ├── parse_data.py          # raw log → 各張表
    ├── local_gate.cu          # 現行 local gate simulator
    ├── cuquantum_baseline.cu  # 逐閘 custatevecApplyMatrix 的 baseline
    ├── bench_gate.cu          # 單閘 microbenchmark（ours vs cuStateVec）
    ├── qiskit_qasm_parser.py  # QASM → .circuit IR
    ├── circuits_native/       # native cp/swap IR
    ├── circuits_ucx/          # u/cx IR
    └── legacy/                # 附錄 B/C 的冷凍舊版（deprecated，不進正常 build）
```

## 執行方法與結果

所有量測都在 `env/Dockerfile` 的容器 image 內執行，主機只提供 GPU driver 與
Docker/Apptainer。建置 image、跑一次最終結果：

```bash
make single-docker     # 建容器 image（CUDA-Q 0.16 + nvcc + nsys/ncu）
make single final      # 自動跑所需的套組，印出報告表 0.1（最終結果）
```

`final` 會自動補跑它依賴的 `suite_ours` 與 `suite_nsys`；資料齊後再下一次
`make single final` 就只是重新組裝印表，不重跑任何量測。

## make 指令

### 表號規則

表號沿用報告的「章節.小節.序號」：正文表 `3.0.1`、`4.1.1`、`7.3.2`…，
附錄表 `B.0.1`、`C.0.3`…，另有 `final`（報告表 0.1）。

### 指令一覽

```bash
make single                  # 只編譯 binary
make single <表號>           # 有資料就只印表；缺資料先跑對應套組再印
make single 7                # 章節縮寫：整章（7.0.1–7.3.2）
make single 4.3              # 小節縮寫：4.3.1–4.3.3
make single B                # 附錄 B 全部
make single-docker           # 建 Docker image
make single-apptainer        # Docker image → SIF
```

修飾參數：

| 參數 | 作用 |
|---|---|
| `SLURM=1` | 套組改用 sbatch 提交（一個套組一個 job，`sbatch -W` 等待）；nsys 的 reduce 與 parse 在提交端做，不佔 GPU |
| `SBATCH_ARGS='--account=...'` | 傳給 sbatch 的額外參數 |
| `RUNNER=docker / apptainer` | 指定容器後端（預設 auto） |
| `IMAGE=<名稱>` | 換 Docker image 名稱 |
| `ARCH=sm_90` | 覆寫 GPU 架構（預設從 nvidia-smi 自動偵測） |
| `ITERS=N` | bench_gate 的迭代次數（預設 50） |
| `APPTAINER_TMPDIR/CACHEDIR` | SIF 建置時的暫存位置 |

### stamp 與解析流程

每張表對應一個 stamp 檔（`results/raw/.suite_*.done` / `.app*.done`），
記錄「資料已量好」：

```
make single <表號>
   │
   ├─ stamp 新鮮（比腳本、binary、電路都新）→ 直接印表
   └─ stamp 缺/過期
        ├─ SLURM=1：sbatch 跑套組（GPU 部分）
        └─ 否則：進容器跑套組
             → raw log（+ nsys trace → CPU 端 reduce）
        → parse_data.py -t <表號> 印表
```

- 強制重跑：刪掉對應的 stamp 檔再下一次指令。
- `parse_data.py` 也可單獨用：`-t list` 列出所有表號、`-t <表號>` 印一張、
  不帶參數則重寫全部 CSV 並更新 `unparsed.txt`（正常應為空；有內容代表某支
  腳本的輸出格式變了）。
- 同一張表跑過多次時 `raw/` 會累積多份 log，parse 一律取**檔名排序最後一份**
  （last run wins）。
- 附錄表的 CSV 由各自腳本直接寫出，不經過 `parse_data.py` 的 raw 解析。

## 表與腳本對照

| 表 | 腳本 | 需求 |
|---|---|---|
| `final` | `suite_ours.sh` + `suite_nsys_*` | GPU + CUDA-Q + nsys |
| 3.0.1 | `suite_ours.sh` | GPU |
| 4.1.1、4.2.1、4.3.1 | `suite_cudaq.sh` | CUDA-Q |
| 4.3.2、4.3.3、7.2.1、7.2.2 | `suite_nsys_trace.sh` + `suite_nsys_reduce.sh` | GPU + CUDA-Q + nsys |
| 6.2.1 | `suite_ours.sh` + `suite_nsys_*` | GPU + CUDA-Q + nsys |
| 7.0.1 | `suite_nsys_*` + `suite_cudaq.sh` | GPU + CUDA-Q + nsys |
| 7.1.1、7.2.3、7.3.1、7.3.2 | `suite_bench_gate.sh` | GPU |
| B.0.1、B.0.2 | `appB_swap.sh` | GPU |
| B.0.3 | `appB.0.3_swap_correctness.sh` | GPU + numpy |
| C.0.1、C.0.3 | `appC_width.sh` | GPU |
| C.0.2 | `appC.0.2_ncu_counters.sh` | GPU + ncu |
| C.0.4 | `appC.0.4_end_to_end.sh` | GPU |

## 測量方法與已知限制

各表的量測邊界（詳細定義見報告 §7.0–7.1）：

- **整電路（3.0.1、6.2.1、7.0.1 的 ours 欄）**：CUDA event 包住 gate 迴圈，
  只算 GPU 執行時間，不含 H2D/D2H、讀檔與正確性 dump。
- **CUDA-Q wall time（4.1.1、4.2.1、4.3.1、7.0.1 的 CUDA-Q 欄）**：
  `cudaq_probe.py` 的 enqueue / materialize+D2H / total，warmup 1、mean of 3。
  4.2.1 的 `*` 欄是 enqueue-only（tensornet 類 backend 無法轉出 dense
  state vector），和 nvidia 的 total 不是同一個邊界。
- **nsys kernel sum（4.3.2、7.2.1、7.2.2）**：on-device 時間，不含 D2H 與
  CPU 啟動；ours 的 trace 內含 1 暖機 + 3 計時所以除以 4，CUDA-Q 的暖機在
  trace 外。7.2.2 是 per-instance 平均，4.3.3 的 instance 數來自單次 trace。
- **單閘 microbenchmark（7.1.1、7.2.3、7.3.1、7.3.2）**：bench_gate 的
  `us/gate`，CUDA event 包 ITERS 次 launch 取平均。

已知限制：

- **7.0.1 的 CUDA-Q 欄**量的是 `get_state` 回傳後剩下的時間（materialize +
  D2H）；`get_state` 在 launch queue 滿時會阻塞，GPU 時間會移進 enqueue，
  所以這一欄的內容會隨負載浮動。嚴格比較請看 7.2.1 的純 GPU kernel sum。
- **4.3.3 的 kernel 特化**在同一個 image 下穩定；不同 cuStateVec/CUDA-Q
  版本可能選到不同特化。
- **C.0.2** 需要 `ncu` 與 performance counter 權限。
- 共享節點繁忙時 wall time 類數字可能膨脹數倍；遇到離群值刪掉對應 stamp
  重跑一次即可。

## 維護注意

- 所有 job 共用 `scripts/common.sh`（`sg_init`/`sg_env`/`sg_build`）。
- 根目錄 `make single` 會 include `single_gate/Makefile`；單閘的 make 規則
  （binary、套組、stamp、表號）都集中在那一個檔案。
- 根目錄 `make tables` 只建 fusion 表，與這裡無關。
- `scripts/legacy/` 是附錄 B/C 的冷凍舊版，標記 deprecated，不進正常 build。
