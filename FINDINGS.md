# DroneDetect — Interim Findings & Verification Closure

> 繁體中文版：[FINDINGS.zh-TW.md](FINDINGS.zh-TW.md) · Project overview: [README.md](README.md)

This document closes the current verification loop: **why PSD + a linear model is
the chosen solution**, the **status of every verification** (with figure links),
and the **conclusions**. Remaining work is additive (§Pending).

## TL;DR

On this dataset, **a normalized 1024-bin Welch PSD fed to a linear classifier (LDA)
is the best-balanced drone-model classifier** — 0.97 segment accuracy across the 7
models, most robust across interference, near-zero training cost. A spectrogram CNN
learns *complementary* cues but never wins on accuracy or robustness. For deployment,
a **single ~25 ms window reaches ~0.95** (≈12.5 ms for ~0.92). The one structural
limit we cannot resolve here is the **model ≡ recording-session confound**.

## Why PSD (and a linear model)

The choice was forced by evidence at each stage, not assumed:

1. **Data quality rules out absolute-amplitude features.** A ~15 dB gain confound
   splits the set into hot (AIR/DIS/PHA) and cold (INS/MIN/MP1/MP2) groups, and
   ~50–60 % of the hot group clips the ADC. So any raw-power feature encodes the
   acquisition, not the drone. Normalizing the PSD (total power → 1) removes the
   gain confound *at the feature level*.
   → [EDA/results/overview_by_drone.png](EDA/results/overview_by_drone.png),
   [EDA/results/box_clip_ratio.png](EDA/results/box_clip_ratio.png)

2. **Spectral shape is almost linearly separable.** LDA on the normalized PSD hits
   0.972 segment / 1.000 recording accuracy; XGBoost does not improve on it, so the
   structure is linear and needs no heavy model. Only MP1↔MP2 (same-family OcuSync)
   meaningfully confuse.
   → [embedding/results/baseline_confusion.png](embedding/results/baseline_confusion.png)

3. **The CNN does not beat it.** A spectrogram CNN scores lower (0.946), *worsens*
   MP1/MP2, and transfers across interference *worse* than LDA (drop 0.19 vs 0.13).
   It does learn different cues (McNemar significant, CKA 0.18, 3-model ensemble
   0.98) — but complementary, not superior.
   → [CNN/results/cnn_confusion.png](CNN/results/cnn_confusion.png)

**Balance scorecard:** accuracy (LDA best), cross-interference robustness (LDA best),
training cost (LDA ~none vs CNN hours on CPU), interpretability (LDA transparent).
PSD + LDA wins on every axis that matters here.

## Verification status

| # | Verification | Result | Figure |
|---|---|---|---|
| 1 | Data-quality audit (gain groups, clipping) | absolute features unusable → normalize | [overview](EDA/results/overview_by_drone.png), [clip](EDA/results/box_clip_ratio.png) |
| 2 | Baseline separability (LDA vs XGBoost, LORO) | 0.972 / 1.000; linear suffices; MP1↔MP2 only confusion | [confusion](embedding/results/baseline_confusion.png) |
| 3 | Interference transfer (LDA) | cross-condition costs ~12–15 pts, never collapses (≥0.75) | [transfer](verify/results/interference_transfer.png) |
| 4 | Model comparison (McNemar, ensemble) | CNN errors differ significantly; ensemble 0.98 > any single | `verify/results/model_comparison.json` |
| 5 | Session-leakage probing + CKA | no run-level fingerprint (PSD run_index 0.05); CKA 0.18 | [probe](verify/results/session_leakage_probe.png) |
| 6 | CNN vs LDA interference transfer | **hypothesis refuted** — CNN transfers worse (drop 0.19 vs 0.13) | [cnn-vs-lda](verify/results/cnn_vs_lda_interference_transfer.png) |
| 7 | Minimum window length | ~12.5 ms → 0.92, ~25 ms → 0.95, diminishing beyond | [sweep](verify/results/segment_length_sweep.png) |
| 8 | Multi-window voting vs. long window | soft > hard; one long window ≥ voting at equal time | [voting](verify/results/multiwindow_voting.png) |

*(LORO = leave-one-run-out; all tests pool the 4 interference conditions unless noted.)*

## Conclusions

- **PSD + linear model is the balanced optimum** — confirmed independently by
  baselines, model comparison, probing, and interference transfer.
- **Deployment window:** with continuous clean observation prefer a single long
  window — **~25 ms ≈ 0.95**, **~12.5 ms ≈ 0.92**. Multi-window voting (use *soft*
  voting) does not beat a long window at equal observation time; it only helps when
  observation is intermittent or a single window may be corrupted.
- **The CNN's role** is a complementary second opinion (ensemble gain), not a
  primary classifier.
- **Hardest residual:** the same-family MP1↔MP2 pair.

## Known limitations

- **Model ≡ session confound is structurally unresolvable here.** Each model was
  likely recorded in one session; leave-one-run-out and the probes only rule out
  the *within-session repeat* fingerprint, not the session identity itself.
  Cross-SDR / cross-day generalisation is unverified.
- **No drone-absent recordings** — supports model *classification*, not presence
  *detection* (that needs external negative samples).
- **256-bin underestimate:** the window-length and voting studies reuse the 256-bin
  spectrogram; native 1024-bin PSD is ~1 pt higher. Trends are unaffected.

## Additive checks

- **Gain-perturbation stress test — done.** Applying ±20 dB of test-time gain, the
  normalized PSD stays flat at 0.96 across the whole range while the un-normalized
  raw log-power collapses (0.51 at −20 dB, 0.60 at +20 dB). Confirms the
  normalization makes the feature gain-invariant, as intended.
  → [verify/results/gain_perturbation.png](verify/results/gain_perturbation.png)
- **Grad-CAM — done.** On the clean-trained CNN, class activation lands on each
  drone's occupied frequency bands, **not** on the DC/LO-leakage line at 0 MHz —
  the model keys on the signal, not a receiver artifact. Each model shows a distinct
  spectral footprint (AIR at ±25 MHz edges, INS a narrow centre band, MP1/MP2 around
  ±5–10 MHz), which is the source of separability; MP1 is still misread as MP2.
  → [CNN/results/gradcam.png](CNN/results/gradcam.png)
- **Higher-frequency-resolution CNN (512 bins) — in progress** — one more attempt to
  close the MP1↔MP2 gap.
