# 執行環境

單閘研究現在**只在容器內執行**。主機只提供 GPU driver、Docker/Apptainer，
以及 repo；CUDA toolkit、nvcc、CUDA-Q、cuQuantum、Python 套件、nsys、ncu
全部由 `single_gate/env/Dockerfile` 的 image 提供。

## Image 內容

| 元件 | 版本 |
|---|---|
| Base | `nvcr.io/nvidia/quantum/cuda-quantum:cu13-0.16.0` |
| OS | Ubuntu 24.04.5 |
| CUDA runtime | 13.0.96 (`cuda-cudart-13-0`) |
| nvcc | 13.0.88 (`cuda-nvcc-13-0`) |
| CUDA-Q | 0.16.0 (cu13) |
| cuQuantum / cuStateVec | 1.14.0 |
| Python | 3.12.3 |
| qiskit / numpy / scipy | 2.6.0 / 2.5.4 / 1.18.1 |
| Nsight Systems | 2025.3.2 |
| Nsight Compute | 2025.3.0 |
| venv | `/opt/single-gate/venv`（`--system-site-packages`）|
| `ARCH` | 預設 `sm_75`；H200/H100 用 `sm_90` |

Image 大小約 **17 GB**（base 12.7 GB + 工具鏈與 venv）。

## 建置與執行

```bash
# repo 根目錄
make single-docker       # docker build -t single-gate:cudaq0.16
make single-apptainer    # docker save -> apptainer build（tar 用完即刪）

# 跑表：make 會自動選 Docker/Apptainer
make single 7.3.2
```

若 Docker image 與 SIF 都不存在，`make single` 只會印出建置指令並停止。
`ncu` 已包含在 image 內，但能不能收集 performance counters 仍取決於執行節點；
Slurm 上若沒有 profiling 權限，C.0.2 會失敗或跳過。

- 不需要設 `PYTHON`、`CUQUANTUM_ROOT`；image 與 Makefile 預設是容器內路徑。
- 不需要 host 的 CUDA-Q SIF；SIF 只作為 Apptainer runner 的 image。
- `ARCH` 未給時，dispatcher 會用 `nvidia-smi` 自動選；H200/H100 也可明確給 `ARCH=sm_90`。
- Slurm：`SBATCH_ARGS="--account=..." make single ID SLURM=1`，會用
  `single_gate/jobs/slurm_container.sh`；partition/GPU/mem/time 都只在這個 wrapper
  宣告，其他 table script 不再有 `#SBATCH`。
- 附錄 B/C 的 legacy build 也全部在 image 內完成。

## Apptainer / SIF

同一份 image 可以轉成 SIF：

```bash
make single-apptainer
# 產生 single_gate/env/single-gate-cudaq0.16.sif

make single 4.3.1 RUNNER=apptainer
```

SIF 固定在 `single_gate/env/single-gate-cudaq0.16.sif`，`make single` 會從這個
位置尋找；中繼 tar 在 repo 根目錄，build 完自動刪除。

image 內路徑與 Docker 完全相同（`/opt/single-gate/venv`、`/usr/local/cuda`、
`/usr/local/lib/python3.12/dist-packages/cuquantum`），所以 script 不用改。

如果使用者的 home 較小，可以在建置時指定 Apptainer 的 cache/tmp：

```bash
make single-apptainer \
  APPTAINER_TMPDIR=/path/to/tmp \
  APPTAINER_CACHEDIR=/path/to/cache
```

不指定時就使用 Apptainer 預設的 `~/.apptainer` 與系統 `/tmp`。
