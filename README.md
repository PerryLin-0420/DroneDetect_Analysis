# DroneDetect 分析專案

將 DroneDetect RF IQ 資料集無損轉換為 parquet，並在其上建立一個 DuckDB summary 層，供無人機訊號的探索式資料分析（EDA）與後續分類研究使用。

## 資料來源

- **Dataset**：DroneDetect Dataset — Radio Frequency Dataset of Unmanned Aerial System (UAS) Signals for Machine Learning
- **作者**：Carolyn J. Swinney, John C. Woods
- **連結**：<https://ieee-dataport.org/open-access/dronedetect-dataset-radio-frequency-dataset-unmanned-aerial-system-uas-signals-machine>
- **DOI**：`10.21227/5jjj-1m32`

本專案僅包含分析程式碼與設計文件；原始資料、轉換後的 parquet、以及 DuckDB summary 檔案都不納入版控（見 [.gitignore](.gitignore)），請自行從上述來源取得資料集。

## 資料集規格（作者提供）

> The DroneDetect dataset consists of 7 different models of popular Unmanned Aerial Systems (UAS) including the new DJI Mavic 2 Air S, DJI Mavic Pro, DJI Mavic Pro 2, DJI Inspire 2, DJI Mavic Mini, DJI Phantom 4 and the Parrot Disco. Recordings were collected using a Nuand BladeRF SDR and using open source software GNURadio. There are 4 subsets of data included in this dataset, the UAS signals in the presence of Bluetooth interference, in the presence of Wi-Fi signals, in the presence of both and with no interference. 3 flight modes are captured - switched on, hovering and flying.

| 項目 | 規格 |
|---|---|
| Sample rate | 60 Mbits/s（= 60,000,000 complex samples/sec） |
| Bandwidth | 28 MHz |
| Centre Freq | 2.4375 GHz |
| 每份錄製長度 | 1.2×10⁸ complex samples（約 2 秒） |
| SDR | Nuand BladeRF |
| 錄製軟體 | GNURadio（open source） |
| 原始格式 | `.dat`，interleaved float32（I, Q 交錯） |

### 無人機型號（7 種）

作者明確列出 7 個型號；下表的「code」欄位為資料集中實際使用的識別碼（來自資料夾與檔名），型號對應除少數明確者外，其餘為依命名合理推斷（作者文件未提供一對一對照表）：

| code（資料夾） | 檔名前綴 | 對應型號 | 對應依據 |
|---|---|---|---|
| `MIN` | MIN | DJI Mavic Mini | 明確 |
| `INS` | INS | DJI Inspire 2 | 明確 |
| `PHA` | PHA | DJI Phantom 4 | 明確 |
| `DIS` | DIS | Parrot Disco | 明確 |
| `AIR` | AIR | DJI Mavic 2 Air S | 推斷（Air） |
| `MP1` | MA1 | DJI Mavic Pro | 推斷（Mavic Pro） |
| `MP2` | MAV | DJI Mavic Pro 2 | 推斷（Mavic Pro 2） |

> 注意：`MP1`/`MP2` 資料夾內的**檔名前綴**分別是 `MA1`/`MAV`，與資料夾名稱不一致。因此 `drone_id` 一律以**資料夾名稱**為準，不可用檔名前綴解析（詳見設計文件「已知資料異常」）。

### 干擾類型（4 種）與飛行模式（3 種）

| 干擾 | 資料夾 | 檔名代碼 `II` |
|---|---|---|
| 無干擾 | `CLEAN` | `00` |
| 僅 Bluetooth | `BLUE` | `01` |
| 僅 Wi-Fi | `WIFI` | `10` |
| Bluetooth + Wi-Fi | `BOTH` | `11` |

| 飛行模式 | 資料夾字尾 | 檔名代碼 `FF` |
|---|---|---|
| Switched on（開機待機） | `_ON` | `00` |
| Hovering（懸停） | `_HO` | `01` |
| Flying（飛行中） | `_FY` | `10` |

### 檔名規則

`<DroneID>_<II><FF>_<RR>.dat`，其中 `RR` 為同一條件下的重複錄製編號（`00`~`04`，共 5 次）。

範例：`MIN_1100_00.dat` = Mavic Mini（`MIN`）＋ Bluetooth+Wi-Fi 干擾（`11`）＋ switched on（`00`）＋ 第 0 次錄製（`00`）。

### 資料集組成與已知缺漏（使用前務必先看）

全資料集共 **390 個 `.dat` 檔**。滿配情況下，每個型號應有 **60 個檔案 = 4 干擾 × 3 飛行模式 × 5 次重複**。但有兩個型號不足 60，做跨型號/跨飛行模式比較時務必注意：

| 型號 | 檔案數 | 說明 |
|---|---|---|
| AIR / INS / MIN / MP1 / MP2 | 各 60 | 滿配（4 干擾 × 3 模式 × 5 run） |
| **DIS**（Parrot Disco） | **40** | 固定翼機無法定點懸停，**沒有 HO（懸停）模式**，只有 ON/FY（4 × 2 × 5）。這是機種物理特性，非資料缺失 |
| **PHA**（Phantom 4） | **50** | 缺 `CLEAN/PHA_FY` 與 `BLUE/PHA_FY` 兩組（各 5 檔）：無干擾與純 Bluetooth 干擾下沒有「飛行中」錄製 |

合計 60×5 + 40 + 50 = **390**。因此：DIS 沒有 HO 可比；PHA 的 FY（飛行）只有 WiFi/Both 兩種干擾基準，無法對照 Clean/Bluetooth。

## 處理流程

### 1. Raw data 轉換（.dat → parquet）

腳本：[load_data_transfer_parquet.py](load_data_transfer_parquet.py)

- **位元級無損**：讀取整個檔案（不截斷）、不做 normalise、`I`/`Q` 保留原始 float32，`zstd` 無損壓縮。
- **來源**：可直接從原始 **zip 壓縮檔**讀取（預設，不需先解壓縮），或從已解壓縮的資料夾讀取。
- **輸出**：鏡像原始資料夾結構，每個 `.dat` 對應一個 `.parquet`（欄位 `index` / `I` / `Q`）。
- **驗證**：[verify_parquet_conversion.py](verify_parquet_conversion.py) 對全部 390 檔做列數比對 + 隨機抽樣 bit-exact 比對，已通過（完整且無損）。

```bash
python load_data_transfer_parquet.py    # 轉換（預設從 zip）
python verify_parquet_conversion.py     # 無損完整性驗證
```

詳細轉換原則與 parquet schema 見設計文件第 2 節。

### 2. Summary DB 建置（parquet → DuckDB）

腳本：`Summary_duckdb/build_summary.py`（本地產物，不進版控）

在 parquet 之上建立**單一 DuckDB 檔案、一張 summary 寬表**（每個錄製一列，共 390 列）：分類 metadata、分布統計、power 特徵、採集端診斷、資料品質檢查全部合併在同一張表。原始 IQ 留在 parquet，db 只存聚合後的 390 列摘要，供 BI/SQL 直接探索。

```bash
python Summary_duckdb/build_summary.py
```

**本 repo 已附上導出的 [Summary_duckdb/summary.parquet](Summary_duckdb/summary.parquet)（約 69 KB、390 列 × 37 欄）**：這是開放格式的可攜成果，不需要備齊 116GB parquet、也不需自行重跑聚合，`pandas`/`DuckDB`/BI 工具都能直接讀取開始探索。本機的 `.duckdb` 與 build 腳本則不納入版控。

完整欄位定義、計算方式與用途，以及後續的 within-run / cross-group 分析邏輯，見設計文件。

## 設計文件

- 繁體中文：[PARQUET_SCHEMA_DESIGN.md](PARQUET_SCHEMA_DESIGN.md)
- English：[PARQUET_SCHEMA_DESIGN.en.md](PARQUET_SCHEMA_DESIGN.en.md)

涵蓋：parquet schema、DuckDB summary 寬表完整 schema、建置流程、分析邏輯、已知資料異常、完整性驗證結果、版控範圍。

## 引用

若使用本資料集，請引用作者原始出處（DOI: `10.21227/5jjj-1m32`）。
