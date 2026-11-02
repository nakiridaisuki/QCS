# 單閘 kernel 研究：實驗筆記（時間線版）

> 這是 SINGLE_GATE_REPORT.md 的實驗筆記版：按探索順序記錄（T1000 → H200 →
> CUDA-Q → 修正 → 再量），保留所有假線索與除錯過程。結論導向的精簡報告見
> 根目錄的 `SINGLE_GATE_REPORT.md`。

# 單閘 kernel 研究：cuStateVec / CUDA-Q vs local gate

> 本報告只涵蓋**單閘 kernel**（不開 gate fusion）。入門、k 的定義、
> 對角/稠密塊策略，以及融合後的分塊數量比較，見 `FUSION_REPORT.md`。

## 摘要

本專案原本用自製的 local-gate 模擬器與 NVIDIA cuStateVec 比較理想態向量模擬。
在 T1000 上兩者速度幾乎完全相同；搬到 H200 後才發現原本是「同撞記憶體頻寬牆」
造成的假象，實際上有輸有贏。進一步把 NVIDIA 的完整工具鏈 CUDA-Q 拉進來當業界
基線後，我們釐清了差距的真正來源：**不是 cuStateVec 的 kernel 比較快，而是電路
分解與特化路線的差別**。為了公平比較，我們補上了兩個原生 gate kernel（受控對角
`cp`、置換 `swap`），讓我們的分解與 CUDA-Q 一致。在**同分解、且排除資料傳輸
（D2H）與 CPU 啟動開銷**後，我們的 backend 在 fp32 快 1.63 倍、fp64 快 1.12 倍。

**表 0.1**　最終結果：兩個分解世代的整電路時間，NV vs ours（H200；單位 ms）。

| 電路 | 分解 | fp32 NV | fp32 ours | fp32 加速 | fp64 NV | fp64 ours | fp64 加速 |
|---|---|---|---|---|---|---|---|
| qft_n20 | u/cx（1000 閘） | 5.93 | 4.09 | **1.45×** | 6.39 | 6.04 | **1.06×** |
| qft_n23 | u/cx（1321 閘） | 45.73 | 37.24 | **1.23×** | 72.03 | 74.17 | 0.97× |
| qft_n25 | u/cx（1561 閘） | 201.79 | 174.04 | **1.16×** | 322.78 | 335.68 | 0.96× |
| qft_n20 | native（220 閘） | 1.03 | 0.46 | **2.25×** | 1.08 | 0.62 | **1.74×** |
| qft_n23 | native（287 閘） | 6.91 | 3.37 | **2.05×** | 8.40 | 6.83 | **1.23×** |
| qft_n25 | native（337 閘） | 30.12 | 18.45 | **1.63×** | 36.88 | 32.95 | **1.12×** |

```bash
make single final   # 自動跑所需的套組（ours + nsys）後組裝；已有資料則純 parse
```

兩段的 NV 欄是不同入口（u/cx 段是我們的 cuStateVec wrapper，native 段是
CUDA-Q fusion off），計時邊界也不同（cudaEvent 閘迴圈 vs nsys kernel sum）；
跨段請勿比絕對值，段內的比值是同邊界的。u/cx 世代有輸有贏（§3），native
世代我們全贏（§7.2）。

---

## 1. 背景

下文路徑皆相對於 repo root：自製模擬器、job 腳本與量測結果在
`single_gate/` 下，CUDA-Q probe 與 `.qasm` 電路分別在 `scripts/baseline/`
與 `benchmarks/`。


| 實作 | 說明 |
|---|---|
| `single_gate/scripts/cuquantum_baseline.cu` | 我們手寫的 cuStateVec wrapper，逐閘呼叫 `custatevecApplyMatrix` |
| `single_gate/scripts/local_gate.cu` | 自製 local-gate 模擬器，逐閘一個 CUDA kernel |
| CUDA-Q `nvidia` | NVIDIA 官方工具鏈的 GPU 態向量後端（底層即 cuStateVec） |
| CUDA-Q `tensornet` / `tensornet-mps` | 張量網路 / MPS，完全不同演算法 |

所有電路都由 Qiskit 讀入 OpenQASM，再轉成統一 IR 後交給兩個自製模擬器。
統一 IR 不是 OpenQASM 的子集，而是一份給 kernel 吃的扁平指令流：

* 第一行是版本字串，接著是 qubit 數、global phase（實部 虛部）、閘數，
  之後每行一個閘。範例：

  ```
  QISKIT_CIRCUIT_V1
  qubits 20
  global_phase 1 0
  gates 220
  <gate 1>
  <gate 2>
  <gate 3>
  ...
  ```

  由 `single_gate/scripts/qiskit_qasm_parser.py` 從 `.qasm` 產生。

* 每個閘只帶 kernel 真正需要的欄位：
    * `u <target> <4 * re/im>`
    * `cx <control> <target>`
    * `cp <control> <target> <re> <im>`
    * `swap <a> <b>`。
* 沒有暫存器、`include`、`measure`、經典條件或參數表達式；那些在 Qiskit
  轉譯時就展開或拒絕。
* OpenQASM 是「電路描述語言」，同一份電路可以有很多種合法寫法；統一 IR 是
  「轉譯後的執行計畫」，閘的種類與順序已經固定，模擬器不需要再解析語法。

輸出態向量後，兩個自製模擬器逐振幅與 cuStateVec 比對（fp32 容差 1e-5、
fp64 1e-12）。CUDA-Q 不匯入 OpenQASM，而是由 `scripts/baseline/cudaq_probe.py`
解析同一份 `.qasm` 後產生 CUDA-Q kernel，詳見 §4。

---

## 2. 第一站：T1000 — 兩者同速

**表 2.0.1**　NVIDIA T1000 8 GB 的硬體資訊。

| 項目 | 值 |
|---|---|
| GPU | NVIDIA T1000 8 GB |
| Compute capability | 7.5（sm_75） |
| SM 數 | 14 |
| CUDA cores | 896 |
| VRAM | 8 GB GDDR6 |
| 記憶體匯流排 | 128-bit |
| 峰值頻寬 | 160 GB/s |
| 實測可用頻寬 | 約 135 GB/s |

在 qft_n20 fp32 上，cuStateVec 與 local gate 分別是 **94.63 ms / 94.07 ms
（差 0.6%）**，幾乎完全一樣，看起來像「同一種實作」。但由 kernel 時間反推的
頻寬都落在 ~135 GB/s（T1000 的實測 DRAM 上限），也就是說**兩者都撞到同一面
記憶體牆**，在這個尺度下實作差異被完全掩蓋。

`nsys` 也顯示兩者 kernel 名稱不同：cuStateVec 走通用的
`constMatApplLoopKernel` / `_Relocate`，我們走自製的 `apply_u_gate_kernel` /
`apply_cx_gate_kernel`。所以並非同一份程式，只是同一個效能瓶頸。

→ 結論：換到頻寬大得多的 GPU 才能看出差別。

---

## 3. 第二站：H200 — 有輸有贏

**表 3.0.0**　NVIDIA H200 的硬體資訊。

| 項目 | 值 |
|---|---|
| GPU | NVIDIA H200 SXM |
| Compute capability | 9.0（sm_90） |
| SM 數 | 132 |
| CUDA cores | 16896 |
| VRAM | 141 GB HBM3e |
| 記憶體匯流排 | 6144-bit |
| 峰值頻寬 | 4.8 TB/s |

### 原始 u/cx 管線：先把電路正規化成 u + cx

Qiskit 以 `optimization_level=0`、`basis_gates=["u","cx"]` 轉譯每個電路，
再把每個 `u` 的 2×2 矩陣、每個 `cx` 的 control/target 寫進統一 IR。兩個
自製模擬器都吃這份 IR；差別只在 backend 一個用 cuStateVec、一個用自製
kernel。量測邊界是「閘迴圈」：CUDA event 包住整串 gate launch，不含電路
載入、初始態設定與最後的 D2H。每個設定先跑 1 次暖機，再計時 3 次取平均。

**表 3.0.1**　u/cx 正規化後的整電路比較（H200；單位 ms，粗體為較快者）。

| 電路 | 閘數 | fp32 cuStateVec | fp32 local gate | fp64 cuStateVec | fp64 local gate |
|---|---|---|---|---|---|
| qft_n20 | 1000 | 5.93 | **4.09** | 6.39 | **6.04** |
| qft_n23 | 1321 | 45.73 | **37.24** | **72.03** | 74.17 |
| qft_n25 | 1561 | 201.79 | **174.04** | **322.78** | 335.68 |


```bash
make single 3.0.1
```

觀察：

* **fp32：local gate 全贏**（16–45%）。
* **fp64：互有勝負**：n20 local 小贏（6.04 vs 6.39 ms），n23/n25 則是
  cuStateVec 小贏（3–4%）。
* 兩者的逐閘時間在同一量級（cuStateVec 的通用 u 約 159 µs；我們的
  `apply_u_gate_kernel` 相近），顯示 wrapper 沒有明顯 per-kernel 懲罰；
  差別在於**用哪些 kernel**，以及 fp32/fp64 的記憶體路徑。

也就是說：在「單閘、無融合」這一層，兩者互有勝負，誰也沒有壓倒性優勢。

---

## 4. 引入 CUDA-Q 作為業界基線

CUDA-Q 0.16 有多個後端 target：

| target | 引擎 |
|---|---|
| `nvidia` / `nvidia-fp64` | cuStateVec（GPU 態向量，fp32/fp64） |
| `tensornet` | cuTensorNet 張量網路 |
| `tensornet-mps` | 矩陣乘積態（MPS） |
| `qpp-cpu` / `density-matrix-cpu` | CPU 態向量 / 密度矩陣 |

### 確認 `nvidia` 後端就是 cuStateVec

CUDA-Q 的 `nvidia` target 由 `libnvqir-custatevec-fp32.so`（fp64 版是
`libnvqir-custatevec-fp64.so`）實作，這個外掛直接連結 `libcustatevec.so.1`。
在 image 內可以用：

```bash
find /opt/nvidia/cudaq -name 'libnvqir-custatevec-*.so'
ldd /opt/nvidia/cudaq/lib/libnvqir-custatevec-fp32.so | grep -E 'custatevec|cuquantum'
```

輸出會看到 `libcustatevec.so.1 => ...`；`nsys` 的 kernel 名稱也全部是
`custatevec::*`。因此 `nvidia` 不是 CUDA-Q 自己重寫的態向量引擎，而是
cuStateVec 的包裝。

CUDA-Q 0.16 **不支援匯入 OpenQASM**（只有 exporter），所以我們用
`scripts/baseline/cudaq_probe.py` 解析同一份 `.qasm`：

1. `parse_qasm()` 讀 `qreg`，把多個暫存器攤平成一個 index space；讀每個閘的
   名稱、參數與 qubit 索引。
2. `gate_source()` 把 QASM 閘映射到 CUDA-Q kernel 呼叫：`h/x/y/z/...` 直接呼叫，
   `cp(λ)` 轉成 `cr1(λ)`，`u(θ,φ,λ)` 轉成 `rz(λ); ry(θ); rz(φ)`。
3. `build_kernel()` 產生一個暫時的 Python 檔，裡面是 `@cudaq.kernel` 裝飾的
   `circuit()`；CUDA-Q 的 decorator 需要真實檔案才能 `inspect` 原始碼。
4. 把 kernel 交給 CUDA-Q 的 compiler/runtime，之後才用 `get_state()` 取得結果。

所以 CUDA-Q 走的是「QASM → Python kernel → CUDA-Q IR → nvidia backend」，
不是「QASM → CUDA-Q 內建 importer」。

### 4.1 CUDA-Q 狀態向量（含 D2H，wall time）

**表 4.1.1**　CUDA-Q `nvidia` 後端 wall time（H200；單位 ms，含 D2H）。

| 電路 | fp32 | fp64 |
|---|---|---|
| qft_n20 | 8.85 | 13.44 |
| qft_n23 | 25.26 | 43.66 |
| qft_n25 | 76.02 | 141.85 |


```bash
make single 4.1.1
```

這張表是 `state prep + D2H`，量測方法與邊界見 §7.1。

（表 4.1.1、表 4.3.1 與表 7.0.1 的 CUDA-Q 欄由同一次量測產生，不再互相矛盾。
注意 wall time 仍有 run 間變異——CPU 排程與 pageable D2H 對節點負載敏感，
曾在繁忙節點量到 2–10 倍膨脹；嚴格比較見 §7.2 的純 GPU 數字。）

### 4.2 張量網路

三個 target 的計時邊界不同（enqueue / materialize / total 的定義與虛擬碼見
§7.0）：`nvidia` 能量到 **total**；tensornet / tensornet-mps 的 state 是張量
網路物件，CUDA-Q 0.16 的 `np.asarray` 與 `to_numpy` 對它們都會直接 raise，
所以這兩個 target 只有 `get_state` 的 **enqueue** 可量（下表以 `*` 標示）。

**表 4.2.1**　CUDA-Q 三種 target 的 wall time（H200；單位 ms；`*` 為 enqueue）。

| 電路 | nvidia fp32 | tensornet | tensornet-mps |
|---|---|---|---|
| qft_n20 | 8.85 | **1.09*** | 1149.51* |


```bash
make single 4.2.1
```

這不是「量測失敗」，而是這兩個 backend 的輸出形式本來就不是 dense vector，
所以沒有 materialize、D2H 與 total 可比。

* **`tensornet` 在 QFT 上光是 enqueue 就比 nvidia 的 enqueue 快**（1.09 vs
  6.31 ms）——這是真正不同的演算法。
* **`tensornet-mps` 在 QFT 上爆掉**（1.15 s）：QFT 的長距離受控相位讓 MPS 的
  bond dimension 爆炸。

張量網路在這裡只用來說明「同一個 CUDA-Q 介面可以換演算法」，不是本報告的
scaling 主軸。

（`*` 欄和 nvidia 欄不是同一個計時邊界；nvidia 的 enqueue 是 6.31 ms。）

### 4.3 gate fusion 開啟 / 關閉

CUDA-Q 的 cuStateVec 後端可做 **gate fusion**（環境變數
`CUDAQ_FUSION_MAX_QUBITS`、`CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS`；設為
非正值即關閉，切到 "direct gate engine"）。

**表 4.3.1**　CUDA-Q gate fusion on/off（H200；單位 ms）。

| 電路 | fp32 fusion on | fp32 fusion off | fp64 fusion on | fp64 fusion off |
|---|---|---|---|---|
| qft_n20 | 8.85 | **7.12** | 13.44 | **9.10** |
| qft_n23 | **25.26** | 27.07 | **43.66** | 44.92 |
| qft_n25 | **76.02** | 98.56 | **141.85** | 170.97 |


```bash
make single 4.3.1
```

（wall time = total，含 D2H；量測規則見 §7.1。）

n20 的 fusion off 反而快約 20–30%，表示融合本身有固定開銷；到 n23/n25 才轉為
正收益（n25 fp32 快約 23%、fp64 快約 17%）。

用 `nsys` 看 GPU kernel 總時間差距更明顯：

**表 4.3.2**　qft_n25 fp32 的 GPU kernel 總時間（H200；單位 ms）。

| 設定 | GPU kernel sum |
|---|---|
| fusion on | ~3.9 |
| fusion off | 30.1 |


```bash
make single 4.3.2
```

量測方法：nsys kernel sum，on-device 時間，不含 D2H 與 CPU 啟動
（規則 3，§7.1）。

**fusion 在 GPU 端快約 8 倍。** kernel 名稱也透露玄機：

**表 4.3.3**　qft_n25 fp32 在 fusion on/off 下的 kernel 家族與 instance 數（H200；nsys）。

| 設定 | kernel 家族 | target 數 | instance | GPU 時間 ms |
|---|---|---|---|---|
| fusion off | `matmulKernel_diagonal<BitExtractor<1>>` | 1 | 300 | 24.241 |
| fusion off | `constMatApplLoopKernel<64,1,…,1>` | 1 | 20 | 3.188 |
| fusion off | `constMatApplLoopKernel<64,1,…,2>` | 2 | 7 | 0.901 |
| fusion off | `constMatApplKernel_Relocate<64,…,1>` | 1 | 5 | 0.797 |
| fusion off | `constMatApplKernel_Relocate<64,…,2>` | 2 | 5 | 0.644 |
| fusion off | `naivePermutingSetKernel` | — | 300 | 0.347 |
| fusion off | `setZeroStateKernel` | — | 1 | 0.001 |
| fusion on | `matmulKernel_diagonal<BitExtractorByWarpVoting>` | 1 | 9 | 1.482 |
| fusion on | `shmemMatApplBatchedKernel<128,5>` | 5 | 4 | 0.832 |
| fusion on | `constMatApplKernel_Relocate<64,…,4>` | 4 | 3 | 0.632 |
| fusion on | `shmemMatApplBatchedRelocateKernel<64,5>` | 5 | 2 | 0.516 |
| fusion on | `constMatApplShTargetsKernel<64,4>` | 4 | 3 | 0.402 |
| fusion on | `naivePermutingSetKernel` | — | 9 | 0.010 |
| fusion on | `setZeroStateKernel` | — | 1 | 0.001 |


```bash
make single 4.3.3
```

（nsys 單次 trace，`--warmup 0 --repeats 1`；規則 3，§7.1。）

「kernel」是 cuStateVec 實際 launch 的 templated GPU 函式；「target 數」是這個
特化版本一次處理幾個 qubit，不是 CPU 端的 target。fusion off 時 CUDA-Q 只能
逐閘呼叫 1–2 target 的通用 kernel；fusion on 後，cuStateVec 把多個閘併成
4–5 target 的矩陣，改走 `constMatApplKernel_Relocate` 與
`shmemMatApplBatched*` 這類針對 shared memory 與 batched 存取最佳化的家族。這正是 cuStateVec 的設計重心：**融合後的多
target、batched、shared-memory 路徑**，而不是單閘的通用路徑。

因此只比較單閘 gate 的 microbenchmark 意義有限：它量的是「最薄的那條路」，
但實際電路會走 fusion；真正該看的是 fusion on/off 後的整電路與 kernel 家族。

---

## 5. 差異分析：CUDA-Q 非融合模式 vs 我們的 local gate

這一章要回答一個問題：既然兩邊都不做 fusion，為什麼 CUDA-Q 還是比較快？
關鍵不是 CUDA-Q 的單閘 kernel 比較強，而是**兩邊收到的電路長得不一樣**。

### 電路怎麼進到兩邊

* **CUDA-Q**：`cudaq_probe.py` 直接讀原始 QASM，保留 `h`、`cp`、`swap` 等閘，
  產生 CUDA-Q kernel。fusion off 時，CUDA-Q 的 compiler 仍然用原生閘的
  語意，閘數是 220 / 287 / 337（n20/23/25）。
* **我們的 local gate**：Qiskit 先 transpile 成 `u` + `cx`，寫成統一 IR；
  每個 `cp` 被拆成約 5 個 `u/cx`，每個 `swap` 拆成 3 個 `cx`。所以同一份
  電路的閘數膨脹成 1000 / 1321 / 1561。若改用 `--native` IR，則保留
  `cp`/`swap`，閘數才會和 CUDA-Q 對齊。

換句話說，前面章節的「閘數不同」不是量測誤差，而是**分解方式不同**。

**表 5.0.1**　同一份電路在三種表示法下的閘數（n20/23/25）。

| 電路 | CUDA-Q 原生 | 我們 u/cx | 我們 native |
|---|---|---|---|
| qft_n20 | 220 | 1000 | 220 |
| qft_n23 | 287 | 1321 | 287 |
| qft_n25 | 337 | 1561 | 337 |

CUDA-Q 原生 = 我們 native：兩邊都是 20/23/25 個 `h`（在我們的 IR 中是 `u`）、
190/253/300 個 `cp`、10/11/12 個 `swap`。u/cx 版本多出來的閘，全部來自
`cp → 5` 與 `swap → 3` 的展開。

### 每個 gate 怎麼拆

| 原閘 | u/cx 分解 | 備註 |
|---|---|---|
| `h` | 1 個 `u` | 任意 1-qubit 么正 |
| `cp(λ)` | 約 5 個 `u`/`cx` | 受控相位被拆成一般 1Q 旋轉 + CNOT |
| `swap(a,b)` | 3 個 `cx` | 標準 SWAP = CX·CX·CX |

這就是為什麼 u/cx 電路的閘數正好是：

* qft_n20：190×5 + 20 + 10×3 = **1000**
* qft_n23：253×5 + 23 + 11×3 = **1321**
* qft_n25：300×5 + 25 + 12×3 = **1561**

---

## 6. 補上兩個原生 gate kernel

### 6.1 四個 gate 在做什麼

以 `n` 個 qubit、little-endian 索引（qubit 0 為最低位）為前提。一個 thread 對應一個
「壓縮索引」——把被操作的 bit 抽掉後的索引，補 0 後得到 `base`，再依局部 pattern
把對應的 bit 設 1，得到實際的態向量 index。

| gate | 作用 | 數學 |
|---|---|---|
| `u` | 任意 1-qubit 么正，作用在 target `t` | 對 `(i0, i1=i0\|2^t)`：`ψ'i0 = u00ψi0 + u01ψi1`，`ψ'i1 = u10ψi0 + u11ψi1` |
| `cx` | CNOT（control `c`、target `t`） | 只有 control=1 半邊：交換 `(c=1,t=0)` 與 `(c=1,t=1)` 兩個振幅 |
| `cp` | 受控相位 `diag(1,1,1,e^{iλ})` | 只有 `(c=1,t=1)` 一個振幅乘上 `e^{iλ}` |
| `swap` | 交換兩個 qubit（a、b） | 交換 `(a=0,b=1)` 與 `(a=1,b=0)`；`\|00⟩`、`\|11⟩` 不動 |

四者其實是同一套規則：對 k 個被操作 bit 的 2^k 維子空間套小矩陣，再依 control
條件選取要處理的局部 pattern。

每個 gate 掃過的態向量倍數：

| gate | 記憶體流量 |
|---|---|
| `u` | 2 × 態向量（讀滿 + 寫滿） |
| `cx` | 1 × 態向量 |
| `swap` | 1 × 態向量 |
| `cp` | 0.5 × 態向量 |

### 6.2 實作變更

這一節補上的不是新演算法，而是「讓我們的分解與 CUDA-Q 一致」所需的原生閘。
改動可以分成四塊：

* **Parser**：`single_gate/scripts/qiskit_qasm_parser.py` 新增 `--native`，
  以 `basis_gates=["u","cx","cp","swap"]`、`optimization_level=0` 轉譯；
  `cp` 與 `swap` 不再被展開。
* **IR**：新增 `cp <control> <target> <re> <im>` 與 `swap <a> <b>` 兩種指令；
  `local_gate.cu` 的 reader 與 `GateType` 同步支援。
* **Kernel**：`local_gate.cu` 新增 `launch_cp_gate` 與 `launch_swap_gate`；
  `swap` 在兩個 bit 都是 chunk 內 lane bit 時走 `apply_swap_gate_lane_kernel`，
  其餘走 scalar kernel（兩條路徑的差異與選擇見附錄 B）；
  `u` 在低位 target 走 warp-transpose/shfl 版本（見附錄 C）。
* **測試**：`single_gate/scripts/legacy/appB/test_swap_kernel.py` 用 numpy 的完整么正矩陣
  驗證 swap 新舊路徑；3.0.1 的電路則逐振幅與 cuStateVec 比對（fp32 容差 1e-5、
  fp64 1e-12）。

新增後，我們的閘數與 CUDA-Q 完全對齊（對照見表 5.0.1）。

**表 6.2.1**　原生 gate 相對 u/cx 的加速幅度（H200；單位 ms；粗體為較快者）。

| 電路 | fp32 u/cx | fp32 native | fp32 加速 | fp64 u/cx | fp64 native | fp64 加速 |
|---|---|---|---|---|---|---|
| qft_n20 | 4.09 | **0.64** | 6.4× | 6.04 | **0.79** | 7.6× |
| qft_n23 | 37.24 | **3.67** | 10.1× | 74.17 | **7.26** | 10.2× |
| qft_n25 | 174.04 | **18.76** | 9.3× | 335.68 | **33.52** | 10.0× |


```bash
make single 6.2.1
```

閘數越多、`cp` 被拆得越碎，加速越明顯：n20 只有 190 個 `cp`，n25 有 300 個。

用記憶體流量做個合理性檢查。qft_n25 fp32 的態向量是 2^25 × 8 B = 268 MB；
cp 讀寫 0.5S、u 讀寫 2S、swap 讀寫 S，總邏輯流量約
`300×0.5S + 25×2S + 12×S = 56.9 GB`。H200 峰值 4.8 TB/s，下限約 11.9 ms；
實測 native fp32 約 19 ms，相當於 63% 有效頻寬。fp64 流量加倍，下限約
23.7 ms，實測約 34 ms（70%）。剩下的是 sector 浪費、transaction 與 kernel
啟動，方向和 §7.3 的 bit 位置分析一致。

---

## 7. 實驗結果：同分解比較

### 7.0 CUDA-Q 的計算流程

CUDA-Q 的 `nvidia` 後端不是「呼叫一次就回傳答案」；`get_state()` 只負責把工作
排進 GPU，真正的計算與複製發生在後面。用虛擬碼表示：

```python
gates = parse_qasm("qft_n25.qasm")
kernel = build_cudaq_kernel(gates)          # QASM -> Python kernel -> CUDA-Q IR

# warm-up：JIT、第一次配置、cache
for _ in range(1):
    s = cudaq.get_state(kernel)
    np.asarray(s)

# 計時：3 次，取平均
for _ in range(3):
    t0 = perf_counter()
    s = cudaq.get_state(kernel)             # enqueue（CPU 排程）
    t1 = perf_counter()
    a = np.asarray(s)                       # materialize + D2H
    t2 = perf_counter()
```

* `enqueue = t1 - t0`：CPU 把 kernel 排進 queue 的時間；GPU 還沒做完。
* `materialize + D2H = t2 - t1`：`np.asarray` 強制 GPU 完成，並把態向量搬回主機。
* `total = t2 - t0`：兩者相加，也就是表 4.1.1 的 wall time。

**為什麼時間一定包含 D2H？** 因為 `cudaq.get_state()` 回傳的是 CUDA-Q 的 state
物件，要拿來和 cuStateVec 逐振幅比對，就必須變成 host 上的 dense array。
`np.asarray` 這一步對 nvidia 後端就是「等 GPU 做完 + cudaMemcpy D2H」。即使
只量 `get_state()`（enqueue），得到的也只是 CPU 排程時間，不是計算時間；所以
本報告的 wall time 一定含 D2H，並在表中明確標示。

**表 7.0.1**　同分解、整電路的 wall time（H200；單位 ms；CUDA-Q 為 fusion off 的 GPU+D2H）。

| 電路 | fp32 ours | fp32 CUDA-Q | fp64 ours | fp64 CUDA-Q |
|---|---|---|---|---|
| qft_n20 | **0.64** | 2.59 | **0.79** | 4.81 |
| qft_n23 | **3.67** | 16.62 | **7.26** | 32.68 |
| qft_n25 | **18.76** | 64.09 | **33.52** | 128.88 |


```bash
make single 7.0.1
```

`ours` 是 native local gate 的 `cudaEvent` 閘迴圈（1 暖機 + 3 次平均）；
`CUDA-Q` 是 fusion off 的 `materialize + D2H`。兩者都不含 CPU 啟動，但
CUDA-Q 多了一次 D2H，所以不是嚴格的同邊界比較；嚴格比較見 §7.2。

### 7.1 計時邊界與 D2H

**D2H = device-to-host copy**，把態向量從 GPU 記憶體搬回主機。我們的模擬器只
計時閘迴圈，D2H 發生在計時結束之後；CUDA-Q 則必須 materialize 才能拿到
可比對的 state（§7.0）。直接拿兩邊的數字相比會嚴重高估差距。

本報告的統一計時規則：

1. **整電路時間（4.1.1、4.2.1、4.3.1、6.2.1、7.0.1）**：1 次暖機 + 3 次計時，
   取 3 次平均。`cudaq_probe.py` 的 enqueue / materialize / total 是同一個
   3 次窗口的 CPU 端 wall time 平均；`local_gate` 的數字是 cudaEvent 包閘迴圈
   的 GPU 時間，同樣在 binary 內做 1 次暖機 + 3 次計時後印平均。
2. **D2H（7.1.1）**：先做 1 次暖機 copy，再量 3 次 `cudaMemcpy` 取平均。
3. **nsys kernel sum（4.3.2、4.3.3、7.2.1、7.2.2）**：kernel sum 是 on-device
   時間，不是 wall time。4.3.2 暖機在 trace 外做，再取 3 個 profiled run 的平均；
   4.3.3/7.2.2 用 `--warmup 0 --repeats 1` 抓單次；7.2.1 每個設定做 3 個
   profiled run 取平均。這類數字不能和 wall time 直接相加。

**表 7.1.1**　D2H 實測（H200；pageable memory，有效頻寬約 14 GB/s；單位 ms）。

| n | fp32 | fp64 |
|---|---|---|
| 20 | 0.84（8.4 MB） | 1.58（16.8 MB） |
| 23 | 4.29（67.1 MB） | 8.34（134.2 MB） |
| 25 | **19.29（268 MB）** | 40.05（537 MB） |


```bash
make single 7.1.1
```

在 n=25 時 D2H（~19 ms fp32）竟然與整個閘迴圈（~19 ms）同量級，所以「含 D2H」
的數字大約被灌了一倍。

### 7.2 修正後的結果（純 GPU）

**表 7.2.1**　原生分解的純 GPU kernel sum（H200；單位 ms；粗體為較快者）。

| 電路 | fp32 ours | fp32 CUDA-Q | fp64 ours | fp64 CUDA-Q |
|---|---|---|---|---|
| qft_n20 | **0.46** | 1.03 | **0.62** | 1.08 |
| qft_n23 | **3.37** | 6.91 | **6.83** | 8.40 |
| qft_n25 | **18.45** | 30.12 | **32.95** | 36.88 |


```bash
make single 7.2.1
```

`ours` 是 native local gate 的 nsys kernel sum；`CUDA-Q` 是 fusion off 的
nsys kernel sum。兩邊都不含 D2H、不含 CPU 啟動。

**表 7.2.2**　qft_n25 逐操作的純 GPU 時間（H200；單位 µs）。

| 操作 | 我們 fp32 | NV fp32 | 我們 fp64 | NV fp64 |
|---|---|---|---|---|
| `cp` × 300 | **46.6** | 164.8 | **81.3** | 90.2 |
| `u` × 25（20 plain + 5 transpose） | **126.9 / 126.4** | 159.4 | **257.2 / 263.5** | 258.0 |
| `swap` × 12 | **108.5** | 128.7 | **175.9** | 255.3 |


```bash
make single 7.2.2
```

（nsys 單次 trace，不含 D2H；規則 3，§7.1。）

**表 7.2.3**　單閘 microbenchmark（H200 n=25 fp32；單位 µs/gate）。

| 操作 | 我們 | NV 最佳相關入口 | 結果 |
|---|---|---|---|
| 通用 1Q（`u`，target=24） | **128.9** | `ApplyMatrix` u = 160.9 | 我們快 1.25× |
| `cp`（ctrl=0, tgt=24） | **66.4** | `ApplyMatrix` cp(diag) = 131.5 | 我們快 1.98× |
| `swap(0,1)` | **143.4** | `Permutation` swap = 179.1 | **我們快 1.25×** |
| `cx`（ctrl=0, tgt=24） | 152.3 | `ApplyMatrix` cx = 131.4 | NV 快 1.16× |


```bash
make single 7.2.3
```

（bench_gate 單閘 microbenchmark：cudaEvent 包 50 次 launch 取平均，µs/gate。）

兩個關鍵觀察：

* `csv_ApplyMatrix` 對對角矩陣**不做特化**：`cp(diag)` 131.5 == `cx` 131.4 == `u` 160.9 附近，
  而 `Permutation` 系列一律較慢（`cp(diag)` 143.2、`cx` 143.2、`swap` 179.1）。
* 這一組每個 gate 只量了一組 qubit 位置，為最差時間 —— `swap` 用的是 `(0,1)`、
  `cp`/`cx` 用的是低位 control。成本其實取決於 bit 位置，見 7.3。

### 7.3 單閘的成本結構：bit 位置掃描

實測顯示成本由**兩個被操作 bit 中較低的那個**決定，不是距離。因為 DRAM 以
32 B sector 為單位：只要 sector 裡有一個振幅被碰到，整個 sector 都會被搬。

**邏輯流量**（理想上必須搬的 bytes，`S = |ψ| = 2^n`）：

| gate | 碰到的振幅 | 讀 | 寫 | 邏輯 bytes |
|---|---|---|---|---|
| `u` | `S` | `S` | `S` | **2S** |
| `cx` | `S/2` | `S/2` | `S/2` | **S** |
| `swap` | `S/2` | `S/2` | `S/2` | **S** |
| `cp` | `S/4` | `S/4` | `S/4` | **S/2** |

（`cx`/`swap` 只碰一半的振幅；`cp` 只碰 `|c=1,t=1⟩` 那 1/4。所謂「swap 是 0.5×」
指的是被碰到的振幅數，「1×」指的是讀+寫的 bytes —— 同一件事的兩種算法。）

**實體流量 = 邏輯流量 ÷ sector 使用率**：

| gate | 兩 bit 都高 | 含低 bit 0 | 兩 bit 同一個 sector |
|---|---|---|---|
| `u` | 2S | 2S | 2S |
| `cx` / `swap` | **S** | **2S** | **2S**（+transaction 開銷） |
| `cp` | **S/2** | **S** | **2S** |

**為什麼 `u` 幾乎沒有這個問題，但 `cx`/`swap`/`cp` 有？**

浪費只發生在「只需要 sector 裡**一部分**振幅」的時候。`u` 需要讀寫**全部** 2^n 顆
振幅，所以每個 sector 本來就是 4/4 全用 —— bit 位置只改變存取的**順序**，
不改變**流量**。反過來，凡是只碰「一半」或「1/4」的 gate，bit 位置就會決定
抓進來的 sector 有沒有被用滿：

| gate | 要抓的 sector 比例 | sector 內使用率 | 實際抓進來 | 真正需要 | 浪費 |
|---|---|---|---|---|---|
| `u` | 全部 | 4/4 | 2S | 2S | 無 |
| `cx` / `swap`（兩 bit 高） | 一半 | 4/4 | S | S | 無 |
| `cx` / `swap`（含低 bit） | 全部 | 2/4 | 2S | S | **2×** |
| `cp`（兩 bit 高） | 1/4 | 4/4 | S/2 | S/2 | 無 |
| `cp`（含低 bit） | 一半 | 2/4 | S | S/2 | **2×** |

所以 `cx` 跟 `swap` 是同構的（都是 1S ↔ 2S），`cp` 只是把 0.5S ↔ 1S
（這也正是 7.3 下面結論 2 講的「`cp` 拿不到它邏輯上的 0.5×」）。

`u` 在 fp64 + 低位 target 曾經掉頻寬；這是附錄 C 的 `wide`/`shfl` 問題，
完整診斷與修正見 C.0.1–C.0.4。下面只列修正後的結果。

**表 7.3.1**　修正後 `u` 的 target 位置掃描（H200；單位 µs/gate）。

| target | fp32 ours | fp32 NV | fp64 ours | fp64 NV |
|---|---|---|---|---|
| t=0 | 128.5 | 161.1 | **286.4**（原 419.4） | 255.3 |
| t=1 | 128.4 | 161.1 | **264.2**（原 330.8） | 255.3 |
| t=2 | 128.5 | 161.0 | **268.9**（原 339.1） | 255.3 |
| t=3 | 128.7 | 161.1 | 268.6（原 268.5） | 255.3 |
| t≥12 | 128.3 – 128.9 | 161.0 – 161.2 | 255.9 – 257.1 | 255.3 – 257.2 |


```bash
make single 7.3.1
```

（測法同表 7.2.3。）

* **fp32 完全是平的**（128.3–128.8 µs，都已在 2S 下限）。
* **fp64 低位原本掉到 419 µs**（t=0），高位 256 µs 才是下限；相鄰 thread 的兩次
  8 B 存取各自只用到半個 sector → **L1 sector 請求量 2×**。已由 `alignas` 合併成
  一次 16 B 存取 + warp 轉置修正，**現在 264–286 µs（t=0 較原值 −32%），斷崖大致
  消除**（t=0 尚有約 8% 殘差）。

**表 7.3.2**　H200 n=25 fp32 的 bit 位置掃描（單位 µs/gate；`swap` 皆為現行 kernel）。

| gate / 位置 | 實測 | 備註 |
|---|---|---|
| `swap(23,24)` | **90** | 最佳（兩 bit 都高） |
| `swap(12,24)` / `swap(12,13)` | 93 / 99 | |
| `swap(2,3)` | 120 | |
| `swap(1,2)` / `swap(0,2)` | 134 / 135 | |
| `swap(0,1)` | 143 | |
| `swap(0,24)`（相距 **24** bit） | 152 | |
| `swap(0,12)` | **158** | 最差 |
| `cp(23,24)` / `cp(12,13)` | **38 / 39** | 邏輯 0.5S 達成 |
| `cp(0,24)` / `cp(0,12)` | 66 / 71 | 含 bit0 → 1× |
| `cp(0,2)` / `cp(0,1)` | 131 / 134 | 兩 bit 同 sector |
| `u`（target=24） | 129 | 2S |
| （對照）NV `Permutation` swap | 179 | 同一張表 |
| （對照）NV `ApplyMatrix` cp(diag) | 131 | 同一張表 |


```bash
make single 7.3.2
```

（測法同表 7.2.3。）

三個結論：

1. **距離無關**：修正後最慢的是 `swap(0,12)`（158 µs），而相距只有 1–2 個 bit 的
   `swap(0,1)` / `swap(0,2)` 反而是 143 / 135 µs。所有 `min(a,b) = 0` 的配對都落在
   135–158 µs 這個區間，跟距離無關；決定成本的是 `min(a,b)` 是否為 0（每個 sector
   只有一半被用到，流量 1S → 2S）。
2. **`cp` 常常拿不到它邏輯上的 0.5×**：表 7.2.3 用的 `ctrl=0` 讓它落到 1×（66.4 µs）；
   只有兩個 bit 都高時才是 0.5×（37.6 µs）。這解釋了 `cp` 在電路內「沒有預期快」。
3. **現行 kernel 已不存在高於流量下限的位置**：最好 90 µs（比 NV 快 2.0×）、
   最差 158 µs（比 NV 的 179 µs 快 1.13×）。先前高於下限的兩個成因
   （「兩 bit 同一個 32 B sector」與「載入寬度不足」）都已修掉。

---

## 8. 已知問題

1. **量測邊界**：D2H 是 pageable memory，數字是上界；nsys 會序列化 kernel，
   kernel sum 只用於相對比較；n=23 fp32 的半個態向量會留在 50 MB L2，這類
   比較請用 n≥25。其餘量測陷阱見附錄 C。
2. **表 7.0.1 的 CUDA-Q 欄**量的是 `get_state` 回傳後剩下的部分
   （materialize + D2H）；`get_state` 在 launch queue 滿時會阻塞，GPU 時間
   會移進 enqueue，所以這欄的內容隨機器負載浮動。跨機器比較請用表 7.2.1
   的純 GPU kernel sum。
3. **節點負載**：共享節點繁忙時 wall time 可膨脹數倍（實測 2–10×）；
   離群值請刪掉對應 stamp 重跑。

---

## 附錄 A：gate 與 NVIDIA API 對應

| 我們的 gate | 語意 | cuStateVec API | NV kernel | 我們的 kernel |
|---|---|---|---|---|
| `u` | 任意 1Q 么正 | `custatevecApplyMatrix`（1 target） | `constMatApplLoopKernel<64,1>` | `apply_u_gate_kernel` |
| `cx` | CNOT | `ApplyMatrix`（X + control）或 `ApplyGeneralizedPermutationMatrix` | `constMatApplLoopKernel` / `naivePermutingSetKernel` | `apply_cx_gate_kernel` |
| `cp` | 受控對角 | `ApplyMatrix`（diag + control）或 generalized permutation | `matmulKernel_diagonal`（CUDA-Q 路徑） | `apply_cp_gate_kernel` |
| `swap` | 2Q 置換 | `ApplyGeneralizedPermutationMatrix`（2 targets） | `naivePermutingSetKernel` | `apply_swap_gate_kernel` |

**方法論注意事項**

* nsys 的 kernel duration 是 on-device 量測，但 profiling 會序列化 kernel；
  只用於**相對**比較，且兩邊條件相同。
* D2H 數字使用 pageable memory，若 CUDA-Q 使用 pinned memory 會更快，故表中
  D2H 為上界。
* 「template 第三參數 = target 數」是從 kernel 名稱與行為推斷，尚待 nTargets
  掃描驗證。

---

## 附錄 B：`swap` kernel 舊 vs 新實作

### 為什麼需要兩套

`swap(a,b)` 交換 `|a=0,b=1⟩` 與 `|a=1,b=0⟩`。實際成本不是由「兩個 bit 相距多遠」
決定，而是由**較低的那個 bit 有多低**決定 —— 因為 DRAM 以 32 B sector 為單位，
只要 sector 裡有一個振幅被碰到，整個 sector 都會被搬（見 7.3）。

每個 32 B chunk 容納 `VEC` 個複數振幅（fp32：4 個，`LOGB=2`；fp64：2 個，
`LOGB=1`），所以索引最低的 `LOGB` 個 bit 是「chunk 內的 lane bit」——
`lane` 指這顆振幅在 chunk 裡排第幾個：`index = (chunk << LOGB) | lane`。
只有當**兩個被操作的 bit 都是 lane bit** 時，整個置換才會留在單一 chunk 內 ——
這是唯一能用一次對齊 32 B 存取做完的情況（fp32 只有 `(0,1)`；fp64 不可能）。

### 三種 bit 位置，與一個 32 B chunk 的關係（以 n=6、fp32、`LOGB=2` 為例）

一個 32 B chunk 容納 4 個複數 = index `4k, 4k+1, 4k+2, 4k+3`，也就是
`|...00⟩ |...01⟩ |...10⟩ |...11⟩`（低兩位是 `q1,q0`）：

| 位置 | 誰在哪個 chunk | 例子 | 結果 |
|---|---|---|---|
| 兩 bit 都是 lane bit | 4 個相位全在同一個 chunk | fp32 `(0,1)` | 一次 32 B load + register 換 lane + 一次 store |
| 一 lane、一 chunk bit | 伙伴在另一個 chunk，但每個 chunk 內只有 2 條 lane 參與 | fp32 `(0,2)` | 需要抓 2 個 chunk → 實測較慢 |
| 兩 bit 都是 chunk bit | 整個 chunk ↔ 另一個整個 chunk，lane 不動 | fp32 `(2,3)`、`(11,12)` | 純量已足夠，向量化無改善 |

### 為什麼「同一個 chunk、一次存取」不等於「最快」

「一次存取」省的是**交易次數**，不是**流量**。DRAM 以 32 B sector 為單位，
只要 sector 裡有一個參與的振幅，整個 sector 就得搬進來。

**表 B.0.1**　以 n=25 fp32（狀態 = 268 MB）為例。

| 位置 | 參與的複數 | 要搬的 sector | 實際 bytes | 實測 | 有效頻寬 |
|---|---|---|---|---|---|
| `swap(23,24)` | 134 MB（一半） | 只有參與的 sector，另一半完全不碰 | **268 MB（1×）** | 91 µs | 2.94 TB/s |
| `swap(0,1)` 新 | 134 MB（一半） | **每一個** sector 都有參與者 | **536 MB（2×）** | 142 µs | 3.77 TB/s |
| `swap(0,1)` 舊 | 同上 | 同上 | **536 MB（2×）** | 255 µs | 2.10 TB/s |


```bash
make single B.0.1
```


* 兩 bit 高：抓到的每個 sector **100% 有用**，但只有一半的記憶體要抓 → 1× 流量。
* 兩 bit 低（同 chunk）：抓到的 sector 只有 **50% 有用**，而且**全部**記憶體都要抓 → 2× 流量。

新 kernel 把有效頻寬從 2.10 拉到 3.77 TB/s（255 → 142 µs），但它**改變不了 2× 的流量**，
所以永遠追不上 1× 的高位情況（91 µs）。

比喻：要把書在兩個書架（sector）之間對調。

* 高位 = 只有一半的書架有書要換 → 只搬那半的書架。
* 低位 = **每個**書架都有 2 本書要換 → 每個書架都得搬下來再放回去。

「一次存取」只是讓你一次抱起整個書架、少跑幾趟；書架本身還是得動。

### 舊：scalar kernel（`apply_swap_gate_kernel`）

一個 thread 對一個 `(a=0,b=0)` base index，用 bit-insertion 算出兩個振幅位置，
做兩次 scalar load + 兩次 scalar store：

```cuda
const uint64_t base    = insert_two_zero_bits(tid, qubit_a, qubit_b);
const uint64_t index01 = base | (1ULL << qubit_b);   // a=0, b=1
const uint64_t index10 = base | (1ULL << qubit_a);   // a=1, b=0
const Complex<T> temp  = state[index01];
state[index01] = state[index10];
state[index10] = temp;
```

問題：相鄰 thread 的 `base` 相差 `2^lo`（`lo = min(a,b)`），所以兩個 stream 的
步幅是 `2^(lo+1)`。當 `lo = 0` 時每次存取只用到 32 B sector 的 1/4，warp 內
產生 4× 的 sector 交易量。

仍保留的原因：這是實測結果，不是「存取次數」的論證 ——

* 兩 bit 都是 chunk bit（`min(a,b) >= LOGB`）時，sector 是「整塊被碰」或「整塊不動」，
  純量版本只搬被碰到的那一半（1× 流量）。向量化的**整塊 chunk 交換**版本流量相同，
  但沒有任何改善（fp32 `(2,3)`：734 µs vs 純量 719 µs，打平）。
* 一 lane、一 chunk bit 的混合情況，**chunk pair** 版本要一次抓兩個 chunk、
  每一條 lane 又都要做條件選擇，反而明顯較慢（fp32 `(0,2)`：1481 vs 917 µs；
  fp64 `(0,1)`：2959 vs 1902 µs）。

所以純量版本留下來，不是因為它「搬得比較少」，而是因為向量化版本在那裡
沒有量到好處、更激進的變體還更慢。

### 新：向量化 intra-chunk kernel（`apply_swap_gate_lane_kernel`）

一個 thread 對一個 32 B chunk：一次對齊 32 B load → 在 register 內換 lane →
一次對齊 32 B store，完全 coalesced：

```cuda
using V = VecChunk<T, LOGB>;                    // alignas(32) { Complex<T> v[1<<LOGB]; }
const V in = *reinterpret_cast<const V*>(state + (chunk << LOGB));
V out;
#pragma unroll
for (int lane = 0; lane < (1 << LOGB); ++lane) {
    const int bit_a = (lane >> qubit_a) & 1;
    const int bit_b = (lane >> qubit_b) & 1;
    out.v[lane] = (bit_a != bit_b) ? in.v[lane ^ lane_mask] : in.v[lane];
}
*reinterpret_cast<V*>(state + (chunk << LOGB)) = out;
```

分派規則（`launch_swap_gate`）：兩個 bit 都 `< LOGB` 走新 kernel，其餘走舊 kernel；
限制也寫在原始碼的 `shortcut:` 註解裡。

### 結果（H200，同一個 bit pair `(0,1)`）

下表兩欄都是相對「完全未修正的 scalar kernel」量出來的（job 612643 與 607658）：

**表 B.0.2**　`swap(0,1)` 的修正前後（相對「完全未修正的 scalar kernel」；job 612643 與 607658）。

| n / precision | 未修正 | 現行 | 變化 | 改善來自 |
|---|---|---|---|---|
| 25 fp32 | 255.4 µs | **142.9 µs** | −44% | lane kernel + `wide` |
| 23 fp32 | 68.1 µs | **38.4 µs** | −44% | lane kernel + `wide` |
| 20 fp32 | 10.76 µs | **6.62 µs** | −38% | lane kernel |
| 25 fp64 | 390.7 µs | **269.6 µs** | −31% | 只有 `wide`（fp64 沒有 lane bit） |
| 23 fp64 | 102.5 µs | **71.1 µs** | −31% | 只有 `wide` |


```bash
make single B.0.2
```


其他 bit pair 只受益於 `wide`（不再有 lane kernel，見 7.3 的掃描）。電路層面：
qft_n25 的 native kernel sum 反而變快（fp32 18.50 → **18.43 ms**、fp64 33.46 → **32.97 ms**）。

### 正確性

`single_gate/scripts/legacy/appB/test_swap_kernel.py` 用 numpy 建立每個 gate 的**完整么正矩陣**算出參考
態向量，再跟 simulator 的輸出比對（電路同時包含 `swap(0,1)` 走新路徑與
`swap(4,5)` 走舊路徑）：

**表 B.0.3**　swap kernel 的正確性（numpy 完整么正矩陣參考）。

| precision | RMSE | 容差 |
|---|---|---|
| fp32 | 3.9e-09 | 1e-05 |
| fp64 | 0 | 1e-12 |


```bash
make single B.0.3
```


---

## 附錄 C：局部 gate kernel 的存取寬度問題與修正

### 症狀

把 1-qubit gate 作用在**低位** target 比高位慢很多，而且只有 fp64（H200 n=25，`u`）：

**表 C.0.1**　`u` 的 target 位置掃描（修正前，H200 n=25，µs）。

| target | 0 | 1 | 2 | 3 | 4 | ≥10 |
|---|---|---|---|---|---|---|
| fp64 `u`（µs） | **419.5** | 330.8 | 338.7 | 268.5 | 265.4 | 255.9 |
| fp32 `u`（µs） | 128.6 | 128.5 | 128.4 | — | — | 128.9 |


```bash
make single C.0.1
```


斷崖落在 t=2→3，之後殘留 ~4% 直到 t≥12；fp32 完全平的；cuStateVec 兩種精度都平
（161 / 255 µs，同一場次量測）。

### 診斷（ncu，計數與時脈無關）

**表 C.0.2**　三種配置的 ncu 計數（n=23）。

| 配置 | L1 ld 請求 | L1 ld sectors | sectors/req | L1 st sectors | L2 read sectors | DRAM read |
|---|---|---|---|---|---|---|
| fp64 t=0 | 524288 | 16777216 | **32** | 16777216 | 4194304 | 134.29 MB |
| fp64 t≥1 | 524288 | 8388608 | 16 | 8388608 | 4194304 | 134.29 MB |
| fp32 t=0 | 524288 | 8388608 | 16 | 8388608 | 2097152 | 67.15 MB |


```bash
make single C.0.2
```


`dram__bytes_*` 與 `lts__t_sectors_*` **在所有配置完全相同** → 排除所有流量解釋。
這張 counters 表是 n=23（134.29 MB = 2^23 × 16 B）；C.0.1 的單閘時間才是 n=25。
唯一差別是 **L1 T-stage 的 sector 請求量**，而 32 sectors/request 是硬體上限
（字面上「每條 thread 各佔一個 sector」）。

根因：**編譯器沒有合併複數的實部與虛部**，SASS 是 `4×LDG.E.64`（fp64）/
`4×LDG.E`（fp32），所以低位 target 時相鄰 thread 落在不同 sector。這個懲罰是
`max(DRAM_time, L1_path_time)`：只有 DRAM 快到讓 L1/LSU 成為瓶頸時才顯現
（T1000 是平的；把 occupancy 壓到 1 blk/SM 也變平）。

### 修正

1. **`wide`（所有 kernel）**：`Complex<T>` 加 `alignas(2 * sizeof(T))`，SASS 變成
   `2×LDG.E.128`（fp64）/ `2×LDG.E.64`（fp32）。一行、零記憶體、零新程式碼路徑。
2. **`shfl`（只有 `u`，target < 5）**：一個 warp 擁有 64 個**連續**振幅，夥伴由
   `__shfl_sync` 取得，所有載入/儲存連續。

### 結果（H200 n=25 fp64 單閘）

**表 C.0.3**　`wide` + `shfl` 修正前後的單閘成本（µs）。
「修正後」用現行 production simulator（`local_gate.cu` 的 `launch_*`，含 shfl），
「修正前」用 legacy `a197aaa` 的 `bench_gate`。

| gate | 修正前 | 修正後 | |
|---|---|---|---|
| `u` t=0 | 416.1 | **263.7** | −36.6% |
| `u` t=1 | 331.3 | **263.9** | −20.4% |
| `u` t=2 | 341.0 | **263.8** | −22.6% |
| `swap(0,2)` | 447.9 | **275.5** | −38.5% |
| `swap(0,1)` | 394.1 | **270.0** | −31.5% |
| `cx(0,2)` | 459.0 | **275.6** | −40.0% |
| `cx(0,1)` | 394.1 | **270.2** | −31.4% |
| `cp(0,12)` | 142.7 | 135.1 | −5.3% |
| `cp(0,2)` | 192.1 | **209.8** | **+9.2%（退步）** |


```bash
make single C.0.3
```


`cp` 在低位退步約 9%（三次重複變異僅 ±0.3%，是真訊號），高位則小幅進步；
端到端仍為淨賺。

### 端到端（同一個 job 內 A/B，兩顆 binary 同源，H200，3 次平均）

**表 C.0.4**　端到端 A/B（H200；單位 ms）。

| 電路 | fp32 前 | fp32 後 | fp64 前 | fp64 後 |
|---|---|---|---|---|
| qft_n20 | 0.901 ms | 0.888 | 1.142 ms | **1.040（−8.9%）** |
| qft_n23 | 3.963 | 3.914 | 7.590 | 7.473 |
| qft_n25 | 19.040 | 19.034 | 34.028 | 33.727 |


```bash
make single C.0.4
```


**全部持平或變快。**

### 其他三個 kernel 的適用性

| kernel | `wide` | warp transpose |
|---|---|---|
| `u` | ✅ 自動 | ✅ 已實作（target < 5） |
| `cx` | ✅ 自動 | ❌ **實測是退步**：它必須碰整個態向量，而 pair kernel 只碰 control=1 半邊。只有 control 低於 sector 大小時才小勝（fp64 control 0：71.2→68.6 µs），不值得為此多一顆 kernel，故不做 |
| `cp` | ✅ 自動 | ❌ 只乘一個振幅，沒有夥伴可交換 |
| `swap` | ✅ 自動 | ❌ 同 `cx` 的理由；且現有的 intra-chunk lane kernel 已經更省 |

### 量測陷阱（供後續參考）

1. **T1000 的記憶體時脈只在持續負載下才會拉滿**。短暫量測達不到峰值頻寬，會產生
   「t=0 特別慢」的**順序假象**，一度讓人誤以為 T1000 重現了此效應。加 ~1 秒熱機後
   T1000 fp64 是完全平的（1933–1940 µs）—— 此效應**只在 H200 這類高頻寬 GPU 出現**。
2. **n=23 fp32 時半個態向量（33.5 MB）會留在 H200 的 50 MB L2**，讓「只碰一半」的
   kernel 看起來特別快。這類比較請用 n ≥ 25。
3. **ncu 預設 `--clock-control base` 會把所有 fp64 配置壓到同一時間**；要相信計數器
   （與時脈無關），不要相信那個時間。
