# DroneDetect 分析專案

> English version: [README.md](README.md)

將 DroneDetect RF IQ 資料集無損轉換為 parquet，在其上建立 DuckDB summary 層，完成探索式資料分析（EDA），並建立以 PSD 為特徵的無人機機型分類 baseline 與魯棒性驗證。最終目標是基於原始 RF 訊號的無人機機型偵測/分類研究。

## 資料來源

- **Dataset**：DroneDetect Dataset — Radio Frequency Dataset of Unmanned Aerial System (UAS) Signals for Machine Learning
- **作者**：Carolyn J. Swinney, John C. Woods
- **連結**：<https://ieee-dataport.org/open-access/dronedetect-dataset-radio-frequency-dataset-unmanned-aerial-system-uas-signals-machine>
- **DOI**：`10.21227/5jjj-1m32`

本專案僅包含分析程式碼、設計文件與小型衍生產物（summary 表、圖表、metrics）；原始資料、轉換後的 parquet、大型特徵檔都不納入版控（見 [.gitignore](.gitignore)），請自行從上述來源取得資料集。

## 資料集規格（作者提供）

| 項目 | 規格 |
|---|---|
| Sample rate | 60 MS/s（complex） |
| Bandwidth | 28 MHz |
| Centre Freq | 2.4375 GHz |
| 每份錄製長度 | 1.2×10⁸ complex samples（約 2 秒） |
| SDR | Nuand BladeRF |
| 錄製軟體 | GNURadio |
| 原始格式 | `.dat`，interleaved float32（I, Q 交錯） |

**7 種無人機型號**（資料夾代碼 → 型號；MP1/MP2 為依命名合理推斷）：

| code（資料夾） | 檔名前綴 | 對應型號 |
|---|---|---|
| `AIR` | AIR | DJI Mavic 2 Air S |
| `DIS` | DIS | Parrot Disco |
| `INS` | INS | DJI Inspire 2 |
| `MIN` | MIN | DJI Mavic Mini |
| `MP1` | **MA1** | DJI Mavic Pro |
| `MP2` | **MAV** | DJI Mavic Pro 2 |
| `PHA` | PHA | DJI Phantom 4 |

> `MP1`/`MP2` 資料夾內的檔名前綴（`MA1`/`MAV`）與資料夾名稱不一致，因此 `drone_id` 一律以**資料夾名稱**為準，不可用檔名前綴解析。

**4 種干擾**——`CLEAN`（00）、`BLUE` Bluetooth（01）、`WIFI`（10）、`BOTH`（11）；**3 種飛行模式**——開機待機 `ON`（00）、懸停 `HO`（01）、飛行 `FY`（10）。檔名規則：`<DroneID>_<II><FF>_<RR>.dat`，`RR` 為重複錄製編號 00~04。

**資料集組成與已知缺漏**（跨型號比較前務必確認）：

| 型號 | 檔案數 | 說明 |
|---|---|---|
| AIR / INS / MIN / MP1 / MP2 | 各 60 | 滿配（4 干擾 × 3 模式 × 5 run） |
| DIS | 40 | 固定翼無法懸停——**沒有 HO 錄製**（機種物理特性，非資料缺失） |
| PHA | 50 | 缺 `CLEAN/PHA_FY` 與 `BLUE/PHA_FY`（無干擾與純 Bluetooth 下沒有飛行錄製） |

合計 **390 檔**。

## 專案結構

```
load_data_transfer_parquet.py   # .dat -> parquet 無損轉換
verify_parquet_conversion.py    # bit-exact 轉換驗證
Summary_duckdb/summary.parquet  # 每錄製一列的 390 列 summary 表（進版控）
EDA/        scripts + results   # summary 特徵的 box plot
embedding/  scripts + results   # 50 ms PSD 特徵 + LDA/XGBoost baseline
CNN/        scripts + results   # spectrogram 萃取 + 小型 2D CNN
verify/     scripts + results   # 魯棒性與模型比較驗證
```

## 處理流程

### 1. Raw data 轉換（.dat → parquet）

[load_data_transfer_parquet.py](load_data_transfer_parquet.py)——位元級無損：整檔讀取、不做 normalise、`I`/`Q` 保留原始 float32、zstd 壓縮、鏡像原始資料夾結構，可直接從原始 zip 讀取。由 [verify_parquet_conversion.py](verify_parquet_conversion.py) 驗證（全部 390 檔列數比對 + 隨機抽樣 bit-exact 比對，已通過）。詳見 [PARQUET_SCHEMA_DESIGN.md](PARQUET_SCHEMA_DESIGN.md)。

### 2. Summary DB 建置（parquet → DuckDB）

`Summary_duckdb/build_summary.py`（本地產物，不進版控）建立單一 390 列寬表：分類 metadata、分布統計、power 特徵、採集端診斷、資料品質欄位（`zero_ratio`、`clip_ratio`）。可攜的 [Summary_duckdb/summary.parquet](Summary_duckdb/summary.parquet) 有進版控。

### 3. EDA（[EDA/](EDA)）

[EDA/scripts/summary_boxplots.py](EDA/scripts/summary_boxplots.py) 對每個 summary 特徵繪製依機型/干擾/飛行模式分組的 box plot 與總覽網格（結果在 `EDA/results/`）。

### 4. PSD embedding + baseline（[embedding/](embedding)）

- [extract_psd_features.py](embedding/scripts/extract_psd_features.py)：每檔切成 40 × 50 ms segment，每段算 1024-bin 雙邊 Welch PSD，總功率正規化為 1（增益不變的頻譜「形狀」）後轉 dB。共 15,591 列。
- [baseline_classify.py](embedding/scripts/baseline_classify.py)：leave-one-run-out CV（以 `run_index` 切 5 fold，同一錄製的 segment 不跨集）、LDA + XGBoost、排除飽和 segment（`clip_ratio > 5%`）。

### 5. Spectrogram CNN（[CNN/](CNN)）

- [extract_spectrograms.py](CNN/scripts/extract_spectrograms.py)：沿用 50 ms segment → STFT（nperseg 1024、hop 512、雙邊），在線性 power 域 mean-pool 到 256(F)×128(T) 網格後轉 dB，存成 float16（約 1 GB，不進版控）。
- [train_cnn.py](CNN/scripts/train_cnn.py)：約 20 萬參數的 4 層 2D CNN、per-segment z-score（去增益，log 域中增益為加性常數）、time-roll + 雜訊 augmentation、leave-one-run-out CV。以 CPU 訓練（無 CUDA GPU；GPU 只影響速度不影響結果）。輸出預測與 128 維 embedding 供比較階段使用。

### 6. 驗證（[verify/](verify)）

- [interference_transfer.py](verify/scripts/interference_transfer.py)：4×4「訓練條件 × 測試條件」準確率矩陣——量化準確率中有多少依賴環境頻譜背景、多少來自無人機訊號本身。
- [model_comparison.py](verify/scripts/model_comparison.py)：對齊 LDA / XGBoost / CNN 的逐 segment 預測，報告 pairwise agreement、McNemar 檢定、獨有答對數、三模型多數決 ensemble——檢驗各模型是否學到互補線索。
- [session_leakage.py](verify/scripts/session_leakage.py)：對 CNN embedding 與 PSD 特徵做線性 probe（GroupKFold 以錄製為單位），預測 `drone_id` / `run_index` / `interference` / `flight_mode`，並計算兩表示的 CKA——檢驗各表示實際編碼了什麼。

## 目前主要發現

### 資料品質

1. **增益 confound（約 15 dB）把資料分成兩群**：AIR/DIS/PHA 錄製增益偏高（平均 −17…−26 dBFS，`max_I` ≈ 0.8–1.0），INS/MIN/MP1/MP2 偏低（−35…−39 dBFS）。這是採集端增益/距離差異，不是機型特性。**任何絕對振幅特徵都被 confound**，必須做 per-recording / per-segment 正規化。
2. **削波（clipping）**：AIR/DIS/PHA 約 50–60% 錄製碰到 ADC full-scale；弱訊號群完全乾淨。兩檔 PHA 嚴重飽和（`BLUE/PHA_ON/PHA_0100_00` 30%、`CLEAN/PHA_ON/PHA_0000_01` 26% 樣本），頻譜分析應剔除。summary 表的 `clip_ratio` 逐檔量化削波程度。
3. 標量統計（`avg_power`、`rms`、`std`）沒有可靠的機型辨識力——被增益 confound 與組內飛行模式變異主導。尺度不變的 `iq_correlation`/`iq_imbalance_db` 較好（2 特徵 RF 5-fold ≈ 0.57），但疑似編碼了 per-session 接收機狀態，故不放入主模型。
4. dB 值在 BI 工具中**絕不可跨錄製做 SUM 或 AVG 聚合**；應先對線性 `avg_power` 平均再轉換（`10·LOG10(AVERAGE(avg_power))`）。

### Baseline 可分性（PSD 形狀，leave-one-run-out）

| 模型 | Segment 準確率 | Recording 準確率（多數決） |
|---|---|---|
| LDA | **0.972 ± 0.004** | **1.000** |
| XGBoost | 0.969 ± 0.006 | 0.987 |

- 7 機型的頻譜形狀近乎線性可分；唯一有意義的混淆是 **MP1 ↔ MP2**（7–8%，同家族 OcuSync 圖傳）。
- 非線性模型在 PSD 特徵上沒有增益——剩餘進步空間在時頻結構。

### Spectrogram CNN vs. PSD baseline

| 模型 | Segment 準確率 | Recording 準確率 |
|---|---|---|
| LDA（PSD，線性） | **0.972** | **1.000** |
| XGBoost（PSD） | 0.969 | 0.987 |
| CNN（spectrogram） | 0.946 | 0.977 |

- CNN **沒有勝過線性 PSD baseline**，且 MP2→MP1 混淆惡化到 16%。最可能原因是頻率解析度：池化後的 spectrogram 只有 256 bins（~234 kHz/bin），PSD 則有 1024 bins（~58.6 kHz/bin），區分同家族機型所需的細頻率結構被池化掉了。
- **但 CNN 學到的是互補資訊，不是劣化版**。McNemar 檢定：CNN vs. 任一 PSD 模型都極顯著（p ≈ 1e-30…1e-37），而 LDA vs. XGBoost 不顯著（p ≈ 0.05）。CNN 獨立答對約 300 個 PSD 模型漏掉的 segment，三模型多數決達 **0.980**，高於任何單一模型。結論：**PSD 頻譜形狀是主判別訊號，時頻結構是次要且正交的補充線索。**

### 干擾遷移魯棒性（LDA）

| train \ test | clean | bluetooth | wifi | both |
|---|---|---|---|---|
| clean | *0.96* | 0.85 | 0.86 | 0.79 |
| bluetooth | 0.84 | *0.98* | 0.85 | 0.84 |
| wifi | 0.80 | 0.75 | *0.98* | 0.91 |
| both | 0.78 | 0.82 | 0.93 | *0.97* |

跨條件遷移掉約 12–15 個百分點但不崩盤：無人機訊號本身在未見過的干擾環境下仍支撐 ≥75% 準確率，其餘 in-distribution 準確率依賴環境頻譜背景。WiFi↔Both 互轉維持高分（皆含 WiFi），證實失效模式是背景頻譜佔用改變。

### Session leakage probing + 表示相似度（CKA）

對各表示做線性 probe（GroupKFold 以錄製為單位），並計算兩者的 CKA：

| Probe 目標 | CNN embedding | PSD features | chance |
|---|---|---|---|
| drone_id（主任務） | 0.95 | 0.97 | 0.16 |
| run_index（洩漏） | *1.00 — artifact* | **0.05** | 0.20 |
| interference | **0.08** | 0.80 | 0.26 |
| flight_mode | 0.50 | 0.80 | 0.36 |

- **訊號中沒有 run-level session 指紋**。有效檢驗是 PSD probe（不經任何模型）：`run_index` 準確率 0.05，*低於* chance——原始頻譜沒有可線性分離的「第幾次重複」指紋。（CNN 的 1.00 是 artifact：embedding 是逐 leave-one-run-out fold 生成的，probe 只是在辨認每個向量出自哪個 fold 的模型，已排除。）
- **CNN 表示對干擾不變，PSD 表示則否**。CNN embedding 幾乎不編碼干擾（0.08，低於 chance），PSD 則強烈編碼（0.80）。這解釋了為何 PSD baseline 在干擾遷移時掉分，並預測 **CNN 的干擾遷移應更魯棒**——這是下一步要驗證的假設。
- **CKA(CNN, PSD) = 0.18**（低）：兩表示確實不同——這是 McNemar 檢定所示互補性的第二個獨立佐證。

### 誠實聲明（caveats）

- **Session confound 在此 dataset 結構上無解**：每個機型很可能是單一 session 錄製，機型 ≡ session。Leave-one-run-out 與上述 probe 都無法排除；run-level probe 只能排除「session 內重複」這一層指紋。跨 SDR / 跨日期泛化未經驗證。
- **沒有「無人機不在場」的錄製**——此 dataset 支援機型*分類*；要做在場*偵測*需要外部負樣本。

## Roadmap

1. ~~無損轉換 + 驗證~~ ✔
2. ~~Summary DB + EDA + 資料品質稽核~~ ✔
3. ~~PSD embedding + 線性/GBM baseline + 干擾遷移檢驗~~ ✔
4. ~~Spectrogram CNN + 預測 agreement/McNemar/ensemble 比較~~ ✔
5. ~~Session leakage probing + CKA~~ ✔（無 run-level 指紋；CNN 表示對干擾不變）
6. **下一步——CNN 干擾遷移矩陣**：對 CNN 跑同樣的 4×4 遷移測試，與 LDA 矩陣對照，驗證 probing 階段得到的「對干擾不變」假設。
7. 後續：spectrogram 的 Grad-CAM 歸因、增益擾動壓力測試，以及（選配）更高頻率解析度的 CNN 重跑以縮小 MP1/MP2 差距。
