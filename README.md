# DroneDetect Analysis Project

> 繁體中文版說明請見 [README.zh-TW.md](README.zh-TW.md)

Lossless conversion of the DroneDetect RF IQ dataset to parquet, a DuckDB summary layer on top of it, exploratory data analysis (EDA), and a PSD-based drone-model classification baseline with robustness verification. The end goal is drone-model detection/classification research on raw RF signals.

## Data source

- **Dataset**: DroneDetect Dataset — Radio Frequency Dataset of Unmanned Aerial System (UAS) Signals for Machine Learning
- **Authors**: Carolyn J. Swinney, John C. Woods
- **Link**: <https://ieee-dataport.org/open-access/dronedetect-dataset-radio-frequency-dataset-unmanned-aerial-system-uas-signals-machine>
- **DOI**: `10.21227/5jjj-1m32`

This repository contains only analysis code, design docs, and small derived artifacts (summary table, plots, metrics). Raw data, converted parquet, and large feature files are not version-controlled (see [.gitignore](.gitignore)); obtain the dataset from the source above.

## Dataset specification (from the authors)

| Item | Spec |
|---|---|
| Sample rate | 60 MS/s (complex) |
| Bandwidth | 28 MHz |
| Centre frequency | 2.4375 GHz |
| Recording length | 1.2×10⁸ complex samples (~2 s) |
| SDR | Nuand BladeRF |
| Recording software | GNURadio |
| Raw format | `.dat`, interleaved float32 (I, Q) |

**7 drone models** (folder code → model; MP1/MP2 mapping inferred from naming):

| Folder code | Filename prefix | Model |
|---|---|---|
| `AIR` | AIR | DJI Mavic 2 Air S |
| `DIS` | DIS | Parrot Disco |
| `INS` | INS | DJI Inspire 2 |
| `MIN` | MIN | DJI Mavic Mini |
| `MP1` | **MA1** | DJI Mavic Pro |
| `MP2` | **MAV** | DJI Mavic Pro 2 |
| `PHA` | PHA | DJI Phantom 4 |

> `MP1`/`MP2` filename prefixes (`MA1`/`MAV`) do not match their folder names. `drone_id` is therefore always parsed from the **folder name**, never the filename prefix.

**4 interference conditions** — `CLEAN` (00), `BLUE` Bluetooth (01), `WIFI` (10), `BOTH` (11) — and **3 flight modes** — switched on `ON` (00), hovering `HO` (01), flying `FY` (10). Filename scheme: `<DroneID>_<II><FF>_<RR>.dat` with `RR` = repeat index 00–04.

**Composition / known gaps** (check before any cross-model comparison):

| Model | Files | Note |
|---|---|---|
| AIR / INS / MIN / MP1 / MP2 | 60 each | full grid (4 interference × 3 modes × 5 runs) |
| DIS | 40 | fixed-wing, cannot hover — **no HO recordings** (by physics, not data loss) |
| PHA | 50 | missing `CLEAN/PHA_FY` and `BLUE/PHA_FY` (no flying recordings without WiFi) |

Total: **390 files**.

## Repository layout

```
load_data_transfer_parquet.py   # .dat -> parquet lossless conversion
verify_parquet_conversion.py    # bit-exact conversion verification
Summary_duckdb/summary.parquet  # 390-row per-recording summary table (committed)
EDA/        scripts + results   # box plots of summary features
embedding/  scripts + results   # 50 ms PSD features + LDA/XGBoost baselines
CNN/        scripts + results   # (next stage) spectrogram CNN
verify/     scripts + results   # robustness & model-comparison checks
```

## Pipeline

### 1. Raw conversion (.dat → parquet)

[load_data_transfer_parquet.py](load_data_transfer_parquet.py) — bit-lossless: full-length reads, no normalisation, raw float32 `I`/`Q`, zstd compression, mirrors the source folder structure. Reads directly from the original zip. Verified by [verify_parquet_conversion.py](verify_parquet_conversion.py) (row counts + random bit-exact sampling on all 390 files, passed). See [PARQUET_SCHEMA_DESIGN.en.md](PARQUET_SCHEMA_DESIGN.en.md).

### 2. Summary DB (parquet → DuckDB)

`Summary_duckdb/build_summary.py` (local, not committed) builds a single 390-row wide table: classification metadata, distribution statistics, power features, acquisition diagnostics, and data-quality columns (`zero_ratio`, `clip_ratio`). The portable export [Summary_duckdb/summary.parquet](Summary_duckdb/summary.parquet) is committed.

### 3. EDA ([EDA/](EDA))

[EDA/scripts/summary_boxplots.py](EDA/scripts/summary_boxplots.py) renders box plots for each summary feature grouped by drone model / interference / flight mode, plus an overview grid (results in `EDA/results/`).

### 4. PSD embedding + baselines ([embedding/](embedding))

- [extract_psd_features.py](embedding/scripts/extract_psd_features.py): slices each recording into 40 × 50 ms segments, computes a 1024-bin two-sided Welch PSD per segment, normalises total power to 1 (gain-invariant spectral shape) then converts to dB. 15,591 segment rows.
- [baseline_classify.py](embedding/scripts/baseline_classify.py): leave-one-run-out CV (5 folds by `run_index`; segments of one recording never straddle folds), LDA + XGBoost, saturated segments (`clip_ratio > 5%`) excluded.

### 5. Verification ([verify/](verify))

- [interference_transfer.py](verify/scripts/interference_transfer.py): 4×4 train-condition × test-condition accuracy matrix — measures how much of the accuracy relies on ambient spectrum context vs. the drone signal itself.

## Key findings so far

### Data quality

1. **Gain confound (~15 dB) splits the dataset into two groups**: AIR/DIS/PHA were recorded hot (avg −17…−26 dBFS, `max_I` ≈ 0.8–1.0) and INS/MIN/MP1/MP2 cold (−35…−39 dBFS). This reflects acquisition gain/distance, not the drones. **Any absolute-amplitude feature is confounded**; per-recording/per-segment normalisation is mandatory.
2. **Clipping**: ~50–60% of AIR/DIS/PHA recordings touch ADC full-scale; the weak-signal group is clean. Two PHA recordings are severely saturated (`BLUE/PHA_ON/PHA_0100_00` 30%, `CLEAN/PHA_ON/PHA_0000_01` 26% of samples) and should be excluded from spectral analysis. `clip_ratio` in the summary table quantifies this per recording.
3. Scalar statistics (`avg_power`, `rms`, `std`) carry no reliable model-discriminative signal — dominated by the gain confound and, within groups, by flight-mode variance. The scale-invariant pair `iq_correlation`/`iq_imbalance_db` separates models better (RF 5-fold ≈ 0.57 with 2 features) but plausibly encodes per-session receiver state, so it is kept out of the main models.
4. dB values must never be SUM- or AVG-aggregated across recordings in BI tools; aggregate the linear `avg_power` first, then convert (`10·LOG10(AVERAGE(avg_power))`).

### Baseline separability (PSD shape, leave-one-run-out)

| Model | Segment acc | Recording acc (majority vote) |
|---|---|---|
| LDA | **0.972 ± 0.004** | **1.000** |
| XGBoost | 0.969 ± 0.006 | 0.987 |

- Spectral shape is almost linearly separable across the 7 models; the only meaningful confusion is **MP1 ↔ MP2** (7–8%, same-family OcuSync downlinks).
- Non-linearity adds nothing on PSD features — the remaining headroom is in time-frequency structure.

### Interference-transfer robustness (LDA)

| train \ test | clean | bluetooth | wifi | both |
|---|---|---|---|---|
| clean | *0.96* | 0.85 | 0.86 | 0.79 |
| bluetooth | 0.84 | *0.98* | 0.85 | 0.84 |
| wifi | 0.80 | 0.75 | *0.98* | 0.91 |
| both | 0.78 | 0.82 | 0.93 | *0.97* |

Cross-condition transfer costs ~12–15 points but never collapses: the drone signal alone supports ≥75% accuracy in unseen interference environments; the rest of the in-distribution accuracy rides on ambient-spectrum context. WiFi↔Both transfer stays high (both contain WiFi), confirming the failure mode is background occupancy change.

### Honest caveats

- **Session confound is structurally unresolvable in this dataset**: each model was likely recorded in one session, so model ≡ session. Leave-one-run-out does not remove it. Cross-SDR / cross-day generalisation is unverified.
- **No drone-absent recordings exist** — the dataset supports model *classification*; building a presence *detector* requires external negative samples.

## Roadmap

1. ~~Lossless conversion + verification~~ ✔
2. ~~Summary DB + EDA + data-quality audit~~ ✔
3. ~~PSD embedding + linear/GBM baselines + interference-transfer check~~ ✔
4. **Next — CNN stage** (`CNN/`): log-magnitude STFT spectrograms (50 ms segments), small 2D CNN targeting the MP1/MP2 confusion and cross-condition robustness.
5. Model comparison (`verify/`): prediction agreement/McNemar, embedding CKA, probing for session/interference leakage, Grad-CAM attribution, gain-perturbation stress test.
