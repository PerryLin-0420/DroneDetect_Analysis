# DroneDetect Parquet 化與探索分析 設計文件

> 英文版：[PARQUET_SCHEMA_DESIGN.en.md](PARQUET_SCHEMA_DESIGN.en.md)

## 1. 背景

原始資料集為 DroneDetect（Carolyn J. Swinney, John C. Woods），RF IQ 錄製資料：
- Sample rate 60 Mbit/s、Bandwidth 28MHz、Centre Freq 2.4375GHz
- 每份錄製 1.2×10⁸ complex samples（約 2 秒）
- 原始格式：`.dat`，interleaved float32（I, Q 交錯）

檔名規則：`<DroneID>_<II><FF>_<RR>.dat`

| 欄位 | 說明 |
|---|---|
| `DroneID` | 無人機型號縮寫：`AIR`/`DIS`/`INS`/`MIN`/`MP1`/`MP2`/`PHA` |
| `II` | 干擾代碼：`00`=clean、`01`=Bluetooth only、`10`=Wi-Fi only、`11`=Bluetooth+Wi-Fi |
| `FF` | 飛行模式代碼：`00`=ON（開機待機）、`01`=HO（懸停）、`10`=FY（飛行中） |
| `RR` | 該條件下第幾次重複錄製（`00`~`04`，共 5 次） |

資料夾結構：`<CLEAN\|BLUE\|WIFI\|BOTH>/<DroneID>_<ON\|HO\|FY>/<DroneID>_<II><FF>_<RR>.dat`

實測全資料集共 **390 個 `.dat` 檔**，大小介於 842MB ~ 960MB（部分錄製時長略短）。

## 2. 轉換 Pipeline（.dat → .parquet）

腳本：[load_data_transfer_parquet.py](load_data_transfer_parquet.py)

輸出根目錄：`D:\DroneEDA\DroneDetect_V2_parquet`（鏡像原始資料夾結構，`.dat` 副檔名換成 `.parquet`）

轉換原則：**位元級無損**
- 讀取整個檔案（不用固定 `count` 截斷，避免丟掉尾端樣本）
- 不做 z-score normalise，`I`/`Q` 為原始 float32 值
- `view(np.complex64)` 僅為 bit 重新詮釋，不改變數值
- 壓縮用 `zstd`（無損，壓縮率優於預設 snappy）
- 已用 bit-exact 比對驗證：讀回的 `I`/`Q` 與原始 `.dat` 逐位元相同

### 2.1 IQ 資料 parquet schema（每個 `.dat` 對應一個 `.parquet`）

| 欄位 | 型別 | 說明 |
|---|---|---|
| `index` | `int64` | 該 sample 在原始檔案中的順序（0-based），用於還原時間順序 |
| `I` | `float32` | In-phase，原始值，未 normalise |
| `Q` | `float32` | Quadrature，原始值，未 normalise |

### 2.2 來源支援：資料夾 或 zip 壓縮檔

`LoadDataTransferParquet` 支援兩種讀取來源，介面統一用「相對於資料集根目錄的 POSIX 路徑字串」（例如 `CLEAN/AIR_ON/AIR_0000_00.dat`），不綁定實體檔案系統路徑：

- **Zip 模式（預設）**：`source_zip` 指向原始 zip 壓縮檔，用 `zipfile` 直接讀取內部成員的 bytes（`ZipFile.read()`），不需要先解壓縮到硬碟。若 zip 內部把整個資料集包在一層外層資料夾（例如 `DroneDetect_V2/CLEAN/...`），會自動偵測並去除該層前綴，確保輸出路徑仍鏡像 `CLEAN/BLUE/WIFI/BOTH` 結構。
- **資料夾模式**：設 `source_zip=None`，`file_folder` 指向已解壓縮的資料夾，用 `rglob("*.dat")` 掃描。

用途：日後如果重新取得這份資料集（原始下載通常是 zip），可以直接從 zip 轉成 parquet，不需要先花時間、多一份硬碟空間解壓縮成 390 個 `.dat` 檔再轉換，避免重工。已用測試 zip（模擬外層包裝資料夾）驗證：zip 模式轉出的 parquet 與資料夾模式的正式輸出逐位元相同。

## 3. DuckDB Summary 層

### 3.1 設計原則：一個 DuckDB、一張集大成的 summary 寬表

在 IQ parquet 之上，建立**單一 DuckDB 檔案**，內含**一張 summary 寬表**，供篩選、分組、跨檔案探索使用。

- **一張寬表**：靜態分類資訊、分布統計、power 特徵、採集端診斷、資料品質檢查，全部合併成同一張表。原因是這些欄位的粒度完全一致——都是「每個檔案一列」，共 **390 列**，用 `relative_path 作為 primary key，沒必要拆多張表再 join。
- **db 裡只存 summary，不存原始 IQ**：1.2 億列 × 390 檔的原始 IQ 留在 parquet；build 腳本用 DuckDB `read_parquet()` 掃描 parquet 算聚合，只把聚合結果（390 列）寫進 `.duckdb`。因此 db 檔很小（數百 KB），是純粹的探索／索引摘要層。
- **探索方便**：BI 工具或 SQL 只查一張 390 列的表，直接 filter/group/pivot，不需要處理 join，也不需要重新掃描 116GB 的 parquet。

### 3.2 Summary 寬表 schema（`summary`，每檔一列，共 390 列）

欄位依用途分組（同一張表）。

**分類 metadata（從路徑/檔名解析，非聚合）**

| 欄位 | 來源 | 說明 |
|---|---|---|
| `relative_path` | 資料夾＋檔名 | 相對於 `DroneDetect_V2_parquet` 的路徑，primary key，回溯原始檔案 |
| `drone_id` | **資料夾名稱**（非檔名前綴，見第 5 節） | `AIR`/`DIS`/`INS`/`MIN`/`MP1`/`MP2`/`PHA` |
| `interference` | 資料夾（`CLEAN`/`BLUE`/`WIFI`/`BOTH`） | 干擾類型（文字） |
| `interference_code` | 檔名 `II` | `00`/`01`/`10`/`11`（數字，方便排序/join） |
| `flight_mode` | 資料夾字尾（`_ON`/`_HO`/`_FY`） | 飛行模式（文字） |
| `flight_mode_code` | 檔名 `FF` | `00`/`01`/`10`（數字） |
| `run_index` | 檔名字尾（`_00`~`_04`） | 同組內第幾次重複錄製，判斷 confound 的關鍵 key |

**基本量與完整性**

| 欄位 | 計算方式 | 用途 |
|---|---|---|
| `sample_count` | `COUNT(*)` | 該檔案樣本數，抓異常短/長的檔案 |
| `duration_sec` | `sample_count / 60_000_000`（作者標示 sample rate：1.2×10⁸ / 2s = 60,000,000 samples/sec） | 實際錄製時長（實測約 1.75s~2.00s） |
| `file_size_parquet` | 輸出 `.parquet` 檔案大小（`stat().st_size`） | ETL 輸出端驗證用，非分析欄位。抓「重跑轉換中途壞掉、寫出殘缺 parquet」；與 `sample_count`（驗證原始讀取完整）針對不同層面 |

**中心趨勢 / DC offset 診斷**

| 欄位 | 計算方式 | 用途 |
|---|---|---|
| `mean_I`, `mean_Q` | `AVG(I)`, `AVG(Q)` | 個別 channel 的 DC bias，判斷是否需要 mean-centering |
| `dc_offset_mag` | `sqrt(mean_I² + mean_Q²)` | 合併成單一數字，快速篩出 DC offset 明顯偏離 0 的檔案 |

**離散度 / scale 診斷**

| 欄位 | 計算方式 | 用途 |
|---|---|---|
| `std_I`, `std_Q` | `STDDEV(I)`, `STDDEV(Q)` | scale 差異；within-run vs cross-group 比較的核心欄位 |
| `p25_I`, `p25_Q` | `approx_quantile(I/Q, 0.25)` | 分布下四分位 |
| `median_I`, `median_Q` | `approx_quantile(I/Q, 0.5)` | 分布中位數，比 mean 抗離群值 |
| `p75_I`, `p75_Q` | `approx_quantile(I/Q, 0.75)` | 分布上四分位 |
| `iqr_I`, `iqr_Q` | `p75 - p25` | 穩健版離散度，與 std 對照可看出 std 是否被離群值拉高 |
| `min_I`, `max_I`, `min_Q`, `max_Q` | `MIN`/`MAX` | 抓 clipping/saturation，檢查 range 合理性 |

**Power（I、Q 聯合計算）**

| 欄位 | 計算方式 | 用途 |
|---|---|---|
| `avg_power` | `AVG(I*I + Q*Q)` | 真實平均功率，判斷飛行模式/型號的 power 差異是訊號還是雜訊 |
| `peak_power` | `MAX(I*I + Q*Q)` | 瞬時最大功率，檢查 ADC clipping |
| `avg_power_db` | `10*log10(avg_power)` | dB scale，方便跨數量級比較與繪圖 |
| `rms_amplitude` | `sqrt(avg_power)` | RF 文獻常用振幅單位；`avg_power` 已算過，只是 `sqrt`，不用額外掃一次 |
| `papr` | `peak_power / avg_power` | Peak-to-Average Power Ratio，衡量訊號突發性，輔助分辨干擾類型 |

**Acquisition Diagnostics（採集端診斷，因果來源待驗證）**

命名上刻意不寫「Receiver / 硬體缺陷」——`iq_imbalance_db`/`iq_correlation` 偏離理想值，可能來自接收端硬體（SDR I/Q demodulator gain/phase mismatch），但也可能來自干擾訊號本身的頻譜特性或其他非接收端因素，不宜在還沒驗證因果之前就預設是硬體壞掉。判斷方法：套用第 4 節的 within-run/cross-group 比較——若這兩個值在所有 drone/interference/flight_mode 組合中都穩定一致，才適合歸因為固定的接收端硬體特性；若隨干擾條件系統性變化，則更可能與訊號本身有關。

| 欄位 | 計算方式 | 用途 |
|---|---|---|
| `iq_imbalance_db` | `20*log10(std_I/std_Q)` | 理想接近 0dB；偏離代表 I/Q 兩通道的 gain 不平衡 |
| `iq_correlation` | `CORR(I, Q)` | 理想接近 0；非 0 代表 quadrature/phase imbalance |

**資料品質檢查**

| 欄位 | 計算方式 | 用途 |
|---|---|---|
| `nan_count` | `COUNT(*) FILTER (isnan(I) OR isnan(Q))` | 抓非數值（NaN），正常原始 ADC 資料不應出現 |
| `inf_count` | `COUNT(*) FILTER (isinf(I) OR isinf(Q))` | 抓無限值，同上 |
| `zero_ratio` | `COUNT(*) FILTER (I=0 AND Q=0) / sample_count` | 抓「整段訊號是死的」這種常見 RF 錄製失敗（例如天線沒接好） |

> `duplicate_ratio`（連續重複值比例，可抓 ADC 卡住不動的硬體故障）概念上有用，但在 1.2 億列/檔的規模下需要 group-by/hash 才能算，成本比上面三個單一數值的聚合高很多。先不放進預設欄位，之後如果 `zero_ratio` 篩出可疑檔案，再針對性地對那幾個檔案額外算即可。

### 3.3 建置流程

腳本：`Summary_duckdb/build_summary.py`（本地產物，不進 git，見第 8 節）

1. 掃描 `DroneDetect_V2_parquet` 底下全部 390 個 `.parquet`
2. 從每個 `relative_path` 解析分類 metadata（`drone_id` 用資料夾名稱）
3. 對每個 parquet 用 DuckDB `read_parquet()` 做**單一 pass 聚合**，算出 3.2 的統計欄位
4. 全部合併成一張 `summary` 表，寫入 `Summary_duckdb/drone_summary.duckdb`

因為每個檔案的統計都是單一 pass 聚合，DuckDB 掃描 parquet 的效率很好；產出的 db 僅 390 列。

## 4. 分析邏輯

利用 `run_index` 維度，將 `summary` 表拆成兩層比較：

1. **Within-group（同 `drone_id`/`interference_code`/`flight_mode_code`，不同 `run_index`）**
   同一條件下 5 次重複錄製理論上訊號特性相同，此組內的變異即為硬體/環境雜訊地板（gain drift、距離、天線角度等）。

2. **Cross-group（不同 `drone_id`/`interference_code`/`flight_mode_code`）**
   若跨組別差異明顯大於 within-group 的雜訊地板，才能判斷該差異是真訊號，可作為分類特徵或決定 normalisation 策略的依據；否則該維度不可靠。

`summary` 僅 390 列，可直接匯入 BI 工具做這兩層比較，不需重新讀取原始 IQ parquet。

## 5. 已知資料異常（完整性驗證時發現）

- **`drone_id` 必須以資料夾名稱為準，不能用檔名前綴解析**：`MP1_*` 資料夾內的檔名前綴實際是 `MA1`（如 `MA1_0110_02.dat`），`MP2_*` 資料夾內的檔名前綴實際是 `MAV`（如 `MAV_0110_01.dat`），與資料夾名稱不一致。其餘 5 種型號（AIR/DIS/INS/MIN/PHA）資料夾與檔名前綴一致。`summary` 表的 `drone_id` 欄位建立時需固定用資料夾名稱，避免誤用檔名前綴分組。
- **`DIS`（Parrot Disco）沒有 HO（懸停）模式**：Disco 是固定翼機（fixed-wing），無法定點懸停，所以只有 `_ON` 與 `_FY` 兩種飛行模式，共 40 列（4 干擾 × 2 模式 × 5 run）。做 flight_mode 比較時，DIS 只有 ON/FY 可比，沒有 HO。
- **`CLEAN/PHA_FY` 與 `BLUE/PHA_FY` 完全缺漏（0 個檔案）**：Phantom 在「無干擾」與「僅 Bluetooth 干擾」下的「飛行中」條件沒有錄製資料，只有 `WIFI/PHA_FY`、`BOTH/PHA_FY` 各 5 個檔案，共 50 列。做跨組別比較時，Phantom 的 FY 條件只能拿到 WiFi/Both 兩種干擾，無法比較 Clean/Bluetooth 基準。
- **各型號列數（已由 summary 驗證）**：AIR/INS/MIN/MP1/MP2 各 60（滿配 4×3×5）、DIS 40、PHA 50，合計 390。滿配以外的缺漏僅上述 DIS 與 PHA 兩處，其餘組合都剛好 5 次重複錄製。

## 6. 完整性驗證結果

驗證腳本：[verify_parquet_conversion.py](verify_parquet_conversion.py)

verify 與轉換腳本共用同一套來源抽象（直接 import `LoadDataTransferParquet`），ground truth 預設從 **zip 壓縮檔**讀取（也可設 `SOURCE_ZIP=None` 改用已解壓縮的 `.dat` 資料夾）。這讓 verify 的生命週期跟轉換流程一致：解壓的 `.dat` 刪除後、只要 zip 還在（或未來重新下載 zip）都能重驗，不會綁死在會被刪除的 `.dat` 資料夾上。

- **列數比對（全部 390 檔，比對 parquet metadata 列數 vs 來源檔案大小/8）**：390/390 通過，無缺檔、無列數不符
- **Bit-exact 抽樣比對（隨機 20 檔，逐位元比對 `I`/`Q`）**：20/20 通過
- **結論：轉換完整且無損**，可以放心基於 `DroneDetect_V2_parquet` 進行後續分析

## 7. 原始 `.dat` 保留政策

**是否刪除 `.dat` 由使用者自行決定並執行**。全數 390 檔已完成轉換與完整性驗證（見第 6 節）。若之後遺失 `.dat` 且需要重新取得，可改用 2.2 節的 zip 模式直接從原始下載檔案重新轉換，不需要再解壓縮一次。刪除 `.dat` 與 zip 之後，`DroneDetect_V2_parquet` 會成為唯一資料副本，刪除前建議先做最後一次完整性驗證（第 6 節）。

## 8. 專案檔案與版控範圍

| 路徑 | 進 git？ | 說明 |
|---|---|---|
| `load_data_transfer_parquet.py` | ✅ | .dat/zip → parquet 轉換 |
| `verify_parquet_conversion.py` | ✅ | 無損完整性驗證 |
| `PARQUET_SCHEMA_DESIGN.md` / `.en.md` | ✅ | 本設計文件（中/英） |
| `.gitignore` | ✅ | — |
| `DroneDetect_V2.zip` | ❌ `*.zip` | 70GB 原始壓縮檔 |
| `DroneDetect_V2/`（`.dat`） | ❌ `*.dat` | 原始資料 |
| `DroneDetect_V2_parquet/` | ❌ | 轉換後 parquet（116GB） |
| `Summary_duckdb/`（build 腳本 + `.duckdb`） | ❌ | DuckDB summary 層，本地產物 |
