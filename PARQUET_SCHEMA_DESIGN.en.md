# DroneDetect Parquet Conversion & Exploratory Analysis — Design Doc

> 中文版：[PARQUET_SCHEMA_DESIGN.md](PARQUET_SCHEMA_DESIGN.md)

## 1. Background

Source dataset: DroneDetect (Carolyn J. Swinney, John C. Woods), RF IQ recordings:
- Sample rate 60 Mbit/s, Bandwidth 28 MHz, Centre Freq 2.4375 GHz
- Each recording is 1.2×10⁸ complex samples (~2 seconds)
- Raw format: `.dat`, interleaved float32 (I, Q interleaved)

Filename convention: `<DroneID>_<II><FF>_<RR>.dat`

| Field | Meaning |
|---|---|
| `DroneID` | Drone model code: `AIR`/`DIS`/`INS`/`MIN`/`MP1`/`MP2`/`PHA` |
| `II` | Interference code: `00`=clean, `01`=Bluetooth only, `10`=Wi-Fi only, `11`=Bluetooth+Wi-Fi |
| `FF` | Flight-mode code: `00`=ON (powered/idle), `01`=HO (hovering), `10`=FY (flying) |
| `RR` | Repeat index for that condition (`00`~`04`, 5 repeats) |

Folder structure: `<CLEAN\|BLUE\|WIFI\|BOTH>/<DroneID>_<ON\|HO\|FY>/<DroneID>_<II><FF>_<RR>.dat`

The full dataset contains **390 `.dat` files**, sized 842 MB ~ 960 MB (some recordings are slightly shorter).

## 2. Conversion Pipeline (.dat → .parquet)

Script: [load_data_transfer_parquet.py](load_data_transfer_parquet.py)

Output root: `D:\DroneEDA\DroneDetect_V2_parquet` (mirrors the source folder structure, `.dat` suffix swapped for `.parquet`)

Conversion principle: **bit-exact lossless**
- Read the entire file (no fixed `count` cap, so no trailing samples are dropped)
- No z-score normalisation; decoded `I`/`Q` are the raw float32 values (bit for bit)
- `view(np.complex64)` is only a byte reinterpretation, values unchanged
- Encoded with the block codec in [iq_codec.py](iq_codec.py), then compressed with `zstd` (level 19); lossless end to end
- Verified bit-exact: decoded `I`/`Q` match the original `.dat` byte for byte

### 2.1 IQ parquet schema (one `.parquet` per `.dat`)

The recorder is a 12-bit ADC: every `I`/`Q` value is exactly `k / 2048` with an integer `k` (observed range about -2059 ~ 2091; true for all 390 files). The common base is therefore the global quantisation step `2048`. The DC offset, by contrast, differs per file (mean 1.6 ~ 186 LSB), so there is no common additive base.

Each channel is cut into blocks of 4096 samples (one parquet row); 256 blocks (1,048,576 samples) form a row group. A block is stored in whichever of two representations has the smaller spread:

- **base mode**: `k - base`, with `base` the rounded block mean
- **delta mode**: first differences, starting from `first` at the block's first sample

The residuals are zigzag-coded to non-negative integers and split into a low-byte plane (`lo`) and a high-byte plane (`hi`). The `hi` plane of a narrow block is all zero and is stored as `NULL`; `zstd` does the entropy coding of the rest.

| Column (`c` is `I` or `Q`) | Type | Description |
|---|---|---|
| `{c}_ref` | `int16` | Base for base mode (0 in delta mode) |
| `{c}_first` | `int16` | `k` of the block's first sample, used to restore delta mode |
| `{c}_mode` | `int8` | 0 = base mode, 1 = delta mode |
| `{c}_lo` | `fixed_size_binary(4096)` | Low bytes of the zigzag residuals |
| `{c}_hi` | `fixed_size_binary(4096)`, nullable | High bytes of the zigzag residuals; `NULL` when all zero |

File metadata carries `iq_codec` (version string) and `iq_samples` (total sample count; the last block is padded with its edge value and trimmed on decode). Parquet column statistics are disabled on write (FLBA min/max would take a large share of the footer).

**Reading**: always use `iq_codec.read_iq(path)` to get float32 `(I, Q)` (or `iq_codec.iter_iq(path)` to stream), not `pq.read_table(columns=["I","Q"])` directly. The codec accepts only finite values that are exactly `k / 2048` (no `-0.0`); for anything else the ETL falls back to plain float32 `I`/`Q` columns (still bit-exact), and `iq_codec` reads both layouts.

**No separate `index` column**: samples are written in the source `.dat` order and parquet preserves row order, so the row position itself is the 0-based sample index. Verification compares `I`/`Q` element-wise against the source `.dat`, so any reordering fails the check.

### 2.2 Source support: folder or zip archive

`LoadDataTransferParquet` supports two read sources through a unified interface — a POSIX-style path string relative to the dataset root (e.g. `CLEAN/AIR_ON/AIR_0000_00.dat`), not tied to a physical filesystem path:

- **Zip mode (default)**: `source_zip` points to the original zip archive; member bytes are read directly via `zipfile` (`ZipFile.read()`), with no need to extract to disk first. If the zip wraps everything in one top-level folder (e.g. `DroneDetect_V2/CLEAN/...`), that prefix is auto-detected and stripped so output paths still mirror `CLEAN/BLUE/WIFI/BOTH`.
- **Folder mode**: set `source_zip=None`; `file_folder` points to the extracted folder, scanned with `rglob("*.dat")`.

Purpose: if the dataset is re-obtained later (the original download is usually a zip), it can be converted straight from the zip — no need to spend time and extra disk space extracting 390 `.dat` files first. Verified with a test zip (simulating a wrapping top-level folder): zip-mode output is bit-exact identical to folder-mode output.

## 3. DuckDB Summary Layer

### 3.1 Design principle: one DuckDB, one all-in-one wide summary table

On top of the IQ parquet, build a **single DuckDB file** containing **one wide summary table** for filtering, grouping and cross-file exploration.

- **One wide table**: classification metadata, distribution stats, power features, acquisition diagnostics and data-quality checks are all merged into the same table. These columns share the exact same granularity — one row per file, **390 rows** total — keyed by `relative_path`, so there is no need to split into multiple tables and join.
- **The db holds only the summary, not the raw IQ**: the 120M-rows × 390-files of raw IQ stay in parquet; the build script uses DuckDB `read_parquet()` to scan the parquet and aggregate, writing only the aggregate result (390 rows) into the `.duckdb`. The db file is therefore tiny (a few hundred KB) — a pure exploration/index summary layer.
- **Easy exploration**: a BI tool or SQL queries a single 390-row table — filter/group/pivot directly, no joins, no need to rescan 49 GB of parquet.

### 3.2 Summary wide-table schema (`summary`, one row per file, 390 rows)

Columns grouped by purpose (all in the same table).

**Classification metadata (parsed from path/filename, not aggregated)**

| Column | Source | Description |
|---|---|---|
| `relative_path` | folder + filename | Path relative to `DroneDetect_V2_parquet`; primary key, traces back to the source file |
| `drone_id` | **folder name** (not filename prefix, see §5) | `AIR`/`DIS`/`INS`/`MIN`/`MP1`/`MP2`/`PHA` |
| `interference` | folder (`CLEAN`/`BLUE`/`WIFI`/`BOTH`) | Interference type (text) |
| `interference_code` | filename `II` | `00`/`01`/`10`/`11` (numeric, for sorting/join) |
| `flight_mode` | folder suffix (`_ON`/`_HO`/`_FY`) | Flight mode (text) |
| `flight_mode_code` | filename `FF` | `00`/`01`/`10` (numeric) |
| `run_index` | filename suffix (`_00`~`_04`) | Repeat index within a group; the key dimension for the confound analysis |

**Basic quantities & completeness**

| Column | Formula | Purpose |
|---|---|---|
| `sample_count` | `COUNT(*)` | Sample count; flags abnormally short/long files |
| `duration_sec` | `sample_count / 60_000_000` (author's sample rate: 1.2×10⁸ / 2s = 60,000,000 samples/sec) | Actual recording length (measured ~1.75s~2.00s) |
| `file_size_parquet` | output `.parquet` size (`stat().st_size`) | ETL output-side validation, not an analysis column. Catches "a re-run broke mid-write and produced a truncated parquet"; complements `sample_count` (which validates source-read completeness) |

**Central tendency / DC offset diagnostics**

| Column | Formula | Purpose |
|---|---|---|
| `mean_I`, `mean_Q` | `AVG(I)`, `AVG(Q)` | Per-channel DC bias; decides whether mean-centering is needed |
| `dc_offset_mag` | `sqrt(mean_I² + mean_Q²)` | Single number, quickly flags files whose DC offset is far from 0 |

**Dispersion / scale diagnostics**

| Column | Formula | Purpose |
|---|---|---|
| `std_I`, `std_Q` | `STDDEV(I)`, `STDDEV(Q)` | Scale differences; the core columns for within-run vs cross-group comparison |
| `p25_I`, `p25_Q` | `approx_quantile(I/Q, 0.25)` | Lower quartile |
| `median_I`, `median_Q` | `approx_quantile(I/Q, 0.5)` | Median, more outlier-robust than mean |
| `p75_I`, `p75_Q` | `approx_quantile(I/Q, 0.75)` | Upper quartile |
| `iqr_I`, `iqr_Q` | `p75 - p25` | Robust dispersion; vs std reveals whether std is inflated by outliers |
| `min_I`, `max_I`, `min_Q`, `max_Q` | `MIN`/`MAX` | Catch clipping/saturation, check range sanity |

**Power (I, Q computed jointly)**

| Column | Formula | Purpose |
|---|---|---|
| `avg_power` | `AVG(I*I + Q*Q)` | True average power; decides whether flight-mode/model power differences are signal or noise |
| `peak_power` | `MAX(I*I + Q*Q)` | Peak instantaneous power; checks ADC clipping |
| `avg_power_db` | `10*log10(avg_power)` | dB scale, for cross-magnitude comparison and plotting |
| `rms_amplitude` | `sqrt(avg_power)` | Amplitude unit common in RF literature; `avg_power` is already computed, just a `sqrt`, no extra scan |
| `papr` | `peak_power / avg_power` | Peak-to-Average Power Ratio; measures burstiness, helps distinguish interference types |

**Acquisition Diagnostics (source of cause to be verified)**

Deliberately not named "Receiver / hardware defect" — `iq_imbalance_db`/`iq_correlation` deviating from ideal may come from the receiver hardware (SDR I/Q demodulator gain/phase mismatch), but may also come from the spectral characteristics of the interference signal itself or other non-receiver factors. Do not assume hardware fault before verifying causality. Method: apply the within-run/cross-group comparison in §4 — if these values are stable across all drone/interference/flight_mode combinations, they can be attributed to a fixed receiver-hardware characteristic; if they vary systematically with interference condition, they more likely relate to the signal itself.

| Column | Formula | Purpose |
|---|---|---|
| `iq_imbalance_db` | `20*log10(std_I/std_Q)` | Ideally ~0 dB; deviation indicates I/Q channel gain imbalance |
| `iq_correlation` | `CORR(I, Q)` | Ideally ~0; non-zero indicates quadrature/phase imbalance |

**Data-quality checks**

| Column | Formula | Purpose |
|---|---|---|
| `nan_count` | `COUNT(*) FILTER (isnan(I) OR isnan(Q))` | Catch NaN, which raw ADC data should never contain |
| `inf_count` | `COUNT(*) FILTER (isinf(I) OR isinf(Q))` | Catch infinities, same |
| `zero_ratio` | `COUNT(*) FILTER (I=0 AND Q=0) / sample_count` | Catch "dead signal" — a common RF recording failure (e.g. antenna not connected) |

> `duplicate_ratio` (consecutive-repeat ratio, catches an ADC stuck on one value) is conceptually useful, but at 120M rows/file it needs a group-by/hash and costs much more than the three single-value aggregates above. Left out of the default columns; if `zero_ratio` flags suspicious files, compute it on just those files.

### 3.3 Build process

Script: `Summary_duckdb/build_summary.py` (local artifact, not in git, see §8)

1. Scan all 390 `.parquet` under `DroneDetect_V2_parquet`
2. Parse classification metadata from each `relative_path` (`drone_id` from the folder name)
3. For each parquet, run a **single-pass aggregation** via DuckDB `read_parquet()` to compute the §3.2 stats
4. Merge everything into one `summary` table, written to `Summary_duckdb/drone_summary.duckdb`

Because each file's stats are a single-pass aggregation, DuckDB scans the parquet efficiently; the resulting db is only 390 rows.

## 4. Analysis Logic

Using the `run_index` dimension, split the `summary` table into two comparison levels:

1. **Within-group (same `drone_id`/`interference_code`/`flight_mode_code`, different `run_index`)**
   The 5 repeats under one condition should have identical signal characteristics in theory, so the variation within this group is the hardware/environment noise floor (gain drift, distance, antenna angle, etc.).

2. **Cross-group (different `drone_id`/`interference_code`/`flight_mode_code`)**
   Only if the cross-group difference is clearly larger than the within-group noise floor can the difference be judged a real signal — usable as a classification feature or to decide a normalisation strategy; otherwise that dimension is unreliable.

`summary` is only 390 rows and can be loaded straight into a BI tool for these two comparisons, with no need to reread the raw IQ parquet.

## 5. Known Data Anomalies (found during integrity verification)

- **`drone_id` must come from the folder name, not the filename prefix**: files inside `MP1_*` folders are actually prefixed `MA1` (e.g. `MA1_0110_02.dat`), and files inside `MP2_*` folders are prefixed `MAV` (e.g. `MAV_0110_01.dat`) — inconsistent with the folder name. The other 5 models (AIR/DIS/INS/MIN/PHA) are consistent. The `summary` table's `drone_id` must be built from the folder name to avoid mis-grouping via the filename prefix.
- **`DIS` (Parrot Disco) has no HO (hovering) mode**: the Disco is fixed-wing and cannot hover, so it only has `_ON` and `_FY` modes — 40 rows (4 interference × 2 modes × 5 runs). For flight_mode comparison, DIS only offers ON/FY, no HO.
- **`CLEAN/PHA_FY` and `BLUE/PHA_FY` are entirely missing (0 files)**: the Phantom has no "flying" recordings under "no interference" or "Bluetooth only" — only `WIFI/PHA_FY` and `BOTH/PHA_FY` exist, 5 files each, 50 rows total. In cross-group comparison, the Phantom FY condition only has WiFi/Both interference and cannot be compared against a Clean/Bluetooth baseline.
- **Row count per model (verified from the summary)**: AIR/INS/MIN/MP1/MP2 have 60 each (full 4×3×5), DIS 40, PHA 50, totalling 390. The only departures from full coverage are the DIS and PHA cases above; every other combination has exactly 5 repeats.

## 6. Integrity Verification Results

Verification script: [verify_parquet_conversion.py](verify_parquet_conversion.py)

verify shares the same source abstraction as the converter (it imports `LoadDataTransferParquet`), reading ground truth by default from the **zip archive** (set `SOURCE_ZIP=None` to verify against the extracted `.dat` folder). This keeps verify's lifecycle aligned with the conversion flow: after the extracted `.dat` files are deleted, as long as the zip exists (or is re-downloaded later) verification still works — it is not tied to the `.dat` folder that gets deleted.

- **Row-count and schema check (all 390 files: `iq_samples` / parquet rows vs source size/8, and columns must be the codec layout or plain float32 `I`/`Q`)**: 390/390 pass, no missing files, no row mismatches
- **Bit-exact spot check (20 random files, decoded `I`/`Q` compared byte for byte, order included)**: 20/20 pass
- **Conclusion: conversion is complete and lossless**, safe to base downstream analysis on `DroneDetect_V2_parquet`

## 7. Raw `.dat` Retention Policy

**Whether to delete the `.dat` files is the user's decision to make and execute.** All 390 files have been converted and integrity-verified (see §6). If the `.dat` files are lost and need to be re-obtained, use the zip mode in §2.2 to reconvert straight from the original download — no re-extraction needed. Once both `.dat` and the zip are deleted, `DroneDetect_V2_parquet` becomes the sole copy of the data; a final integrity verification (§6) is recommended before deleting.

## 8. Project Files & Version-Control Scope

| Path | In git? | Notes |
|---|---|---|
| `load_data_transfer_parquet.py` | ✅ | .dat/zip → parquet conversion |
| `iq_codec.py` | ✅ | Lossless block codec for the IQ parquet files (encode / decode / read / write) |
| `verify_parquet_conversion.py` | ✅ | Lossless integrity verification |
| `PARQUET_SCHEMA_DESIGN.md` / `.en.md` | ✅ | This design doc (zh/en) |
| `.gitignore` | ✅ | — |
| `DroneDetect_V2.zip` | ❌ `*.zip` | 70 GB raw archive |
| `DroneDetect_V2/` (`.dat`) | ❌ `*.dat` | Raw data |
| `DroneDetect_V2_parquet/` | ❌ | Converted parquet (~49 GB) |
| `Summary_duckdb/` (build script + `.duckdb`) | ❌ | DuckDB summary layer, local artifact |
