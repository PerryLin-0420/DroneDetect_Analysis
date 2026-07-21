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
CNN/        scripts + results   # spectrogram extraction + small 2D CNN
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

### 5. Spectrogram CNN ([CNN/](CNN))

- [extract_spectrograms.py](CNN/scripts/extract_spectrograms.py): same 50 ms segments → STFT (nperseg 1024, hop 512, two-sided), mean-pooled in the linear power domain to a 256(F)×128(T) grid, then dB, stored as float16 (~1 GB, not committed).
- [train_cnn.py](CNN/scripts/train_cnn.py): ~200k-param 4-block 2D CNN, per-segment z-score (removes gain — additive in log domain), time-roll + noise augmentation, leave-one-run-out CV. CPU-trained (no CUDA GPU present; GPU only changes speed, not results). Exports predictions + 128-d embeddings for the comparison stage.

### 6. Verification ([verify/](verify))

- [interference_transfer.py](verify/scripts/interference_transfer.py): 4×4 train-condition × test-condition accuracy matrix — measures how much of the accuracy relies on ambient spectrum context vs. the drone signal itself.
- [model_comparison.py](verify/scripts/model_comparison.py): aligns LDA / XGBoost / CNN per-segment predictions, reports pairwise agreement, McNemar's test, exclusive-correct counts, and a 3-model majority-vote ensemble — tests whether the models learned complementary cues.
- [session_leakage.py](verify/scripts/session_leakage.py): linear probes (GroupKFold by recording) on CNN embeddings and PSD features for `drone_id` / `run_index` / `interference` / `flight_mode`, plus CKA between the two representations — tests what each representation actually encodes.
- [cnn_interference_transfer.py](verify/scripts/cnn_interference_transfer.py): trains one model per interference condition (runs 0–3) and tests on every condition, for both CNN and LDA under one protocol — a fair CNN-vs-PSD cross-interference robustness comparison. Saves the 4 CNN weights to `CNN/models/`.

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

### Spectrogram CNN vs. PSD baseline

| Model | Segment acc | Recording acc |
|---|---|---|
| LDA (PSD, linear) | **0.972** | **1.000** |
| XGBoost (PSD) | 0.969 | 0.987 |
| CNN (spectrogram) | 0.946 | 0.977 |

- The CNN **does not beat the linear PSD baseline**, and MP2→MP1 confusion actually worsens to 16%. The most likely cause is frequency resolution: the pooled spectrogram has 256 bins (~234 kHz/bin) vs. the PSD's 1024 bins (~58.6 kHz/bin), so the fine spectral detail that separates same-family models was pooled away.
- **But the CNN learns complementary information, not a degraded copy.** McNemar's test: CNN vs. either PSD model is highly significant (p ≈ 1e-30…1e-37), while LDA vs. XGBoost is not (p ≈ 0.05). The CNN is exclusively right on ~300 segments the PSD models miss, and a 3-model majority vote reaches **0.980** — above any single model. Conclusion: **PSD spectral shape is the primary discriminative signal; time-frequency structure is a secondary, orthogonal cue.**

### Interference-transfer robustness (LDA)

| train \ test | clean | bluetooth | wifi | both |
|---|---|---|---|---|
| clean | *0.96* | 0.85 | 0.86 | 0.79 |
| bluetooth | 0.84 | *0.98* | 0.85 | 0.84 |
| wifi | 0.80 | 0.75 | *0.98* | 0.91 |
| both | 0.78 | 0.82 | 0.93 | *0.97* |

Cross-condition transfer costs ~12–15 points but never collapses: the drone signal alone supports ≥75% accuracy in unseen interference environments; the rest of the in-distribution accuracy rides on ambient-spectrum context. WiFi↔Both transfer stays high (both contain WiFi), confirming the failure mode is background occupancy change.

### Session-leakage probing + representation similarity (CKA)

Linear probes (GroupKFold by recording) on each representation, and CKA between them:

| Probe target | CNN embedding | PSD features | chance |
|---|---|---|---|
| drone_id (task) | 0.95 | 0.97 | 0.16 |
| run_index (leakage) | *1.00 — artifact* | **0.05** | 0.20 |
| interference | **0.08** | 0.80 | 0.26 |
| flight_mode | 0.50 | 0.80 | 0.36 |

- **No run-level session fingerprint in the signal.** The valid test is the PSD probe (touches no model): `run_index` accuracy is 0.05, *below* chance — the raw spectrum carries no linearly separable "which repeat" fingerprint. (The CNN's 1.00 is an artifact: its embedding is generated per leave-one-run-out fold, so the probe merely recovers which fold's model emitted each vector. It is excluded.)
- **The CNN embedding used here barely encodes interference (0.08, below chance) while PSD strongly does (0.80).** Note this embedding comes from the leave-one-run-out models, which *saw all four interference conditions* during training — its invariance is a product of mixed-interference training. This led to a hypothesis (CNN transfers more robustly) that the transfer experiment below then **refuted** — see the caveat there.
- **CKA(CNN, PSD) = 0.18** (low): the two representations are genuinely different — a second independent confirmation of the complementarity that McNemar's test showed.

### Interference transfer: CNN vs. PSD (unified protocol) — hypothesis refuted

Train one model per condition (runs 0–3); diagonal tests the held-out run 4 of the same condition, off-diagonal tests the other conditions:

| | on-diagonal (held-out) | off-diagonal (cross-interference) | drop |
|---|---|---|---|
| LDA (PSD) | 0.96 | 0.83 | **0.13** |
| CNN (spectrogram) | 0.91 | 0.72 | **0.19** |

- **The CNN transfers *worse*, not better** — larger drop and lower accuracy in every off-diagonal cell. The probing-stage prediction is refuted.
- **Why the prediction failed:** the probe's interference-invariance came from an embedding trained on *all* interference conditions. The transfer models are trained on a *single* condition each, a different setup. With only ~3k segments per condition, the high-capacity CNN overfits the training condition's spectral background, while the low-capacity linear LDA generalises across conditions better. This is the classic "small data + out-of-distribution → simpler model wins" pattern, and it reinforces the project's through-line: **on this dataset, PSD + a linear model is the strongest and most robust combination.**

### Honest caveats

- **Session confound is structurally unresolvable in this dataset**: each model was likely recorded in one session, so model ≡ session. Leave-one-run-out and the probes above cannot remove it; the run-level probe only rules out the *within-session repeat* fingerprint. Cross-SDR / cross-day generalisation is unverified.
- **No drone-absent recordings exist** — the dataset supports model *classification*; building a presence *detector* requires external negative samples.

## Roadmap

1. ~~Lossless conversion + verification~~ ✔
2. ~~Summary DB + EDA + data-quality audit~~ ✔
3. ~~PSD embedding + linear/GBM baselines + interference-transfer check~~ ✔
4. ~~Spectrogram CNN + prediction agreement/McNemar/ensemble comparison~~ ✔
5. ~~Session-leakage probing + CKA~~ ✔ (no run-level fingerprint)
6. ~~CNN interference-transfer matrix vs. LDA~~ ✔ (hypothesis refuted: single-condition CNN transfers worse than linear PSD baseline)
7. Remaining: Grad-CAM attribution on spectrograms, gain-perturbation stress test, and (optional) higher-frequency-resolution CNN re-run to close the MP1/MP2 gap.

**Overall conclusion so far:** across every stage — baselines, model comparison, probing, and interference transfer — PSD spectral shape with a linear/low-capacity model is the strongest, most robust drone-model classifier on this dataset. The CNN learns complementary cues (significant McNemar, low CKA, ensemble gain) but does not win on accuracy or cross-interference robustness. The hardest residual is the same-family MP1↔MP2 pair. The deployment-level generalisation question (model ≡ session) remains structurally unanswerable here.
