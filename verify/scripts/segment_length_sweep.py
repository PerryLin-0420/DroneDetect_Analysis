### How short can the observation window be and still identify the drone? ###
### The 50 ms pooled spectrogram (256F x 128T) has one PSD column per        ###
### 0.39 ms; averaging k consecutive columns IS the Welch PSD of a k*0.39 ms  ###
### window. So we sweep window length by slicing the time axis -- no need to  ###
### re-read the 116 GB of raw IQ.                                             ###
### For each length: leave-one-run-out LDA, and at test time draw several     ###
### random-start windows per recording (Monte Carlo) to get an accuracy       ###
### distribution. All interference conditions are pooled.                     ###
### Frequency resolution is 256 bins (from the spectrogram), coarser than the ###
### native 1024-bin PSD, so absolute accuracy is a slight underestimate but   ###
### the length trend is reliable.                                            ###

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

sys.stdout.reconfigure(line_buffering=True)
RNG = np.random.default_rng(0)

SCRIPT_DIR = Path(__file__).resolve().parent
SPECS_NPY = SCRIPT_DIR / ".." / ".." / "CNN" / "results" / "spectrograms.npy"
SPEC_META = SCRIPT_DIR / ".." / ".." / "CNN" / "results" / "spectrogram_meta.parquet"
RESULTS_DIR = SCRIPT_DIR / ".." / "results"

DRONE_ORDER = ["AIR", "DIS", "INS", "MIN", "MP1", "MP2", "PHA"]
BAD_CLIP = 0.05
TOTAL_FRAMES = 128
MS_PER_FRAME = 50.0 / TOTAL_FRAMES               # 0.390625 ms per spectrogram column
K_FRAMES = [1, 2, 4, 8, 16, 32, 64, 128]         # window lengths to sweep
N_MC = 10                                        # random-start windows per test recording

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
C_LINE, C_BAND, C_REF = "#2a78d6", "#9ec5f4", "#898781"


def window_features(specs, k, rng):
    """One random k-frame window per spectrogram -> (N, 256) z-scored PSD."""
    n = specs.shape[0]
    starts = rng.integers(0, TOTAL_FRAMES - k + 1, size=n) if k < TOTAL_FRAMES else np.zeros(n, int)
    feats = np.empty((n, specs.shape[1]), dtype=np.float32)
    for i in range(n):
        feats[i] = specs[i, :, starts[i]:starts[i] + k].mean(axis=1)
    # per-vector z-score across frequency: removes the per-recording gain offset
    mu = feats.mean(axis=1, keepdims=True)
    sd = feats.std(axis=1, keepdims=True) + 1e-6
    return (feats - mu) / sd


def sweep(specs, meta):
    y = np.array([DRONE_ORDER.index(d) for d in meta["drone_id"]])
    runs = sorted(meta["run_index"].unique())
    results = {}
    for k in K_FRAMES:
        ms = k * MS_PER_FRAME
        fold_means = []
        for fold in runs:
            te = (meta["run_index"] == fold).to_numpy()
            # train on one random window per training recording
            Xtr = window_features(specs[~te], k, RNG)
            clf = LinearDiscriminantAnalysis().fit(Xtr, y[~te])
            # test: N_MC independent random windows per test recording
            mc_acc = []
            for _ in range(N_MC):
                Xte = window_features(specs[te], k, RNG)
                mc_acc.append(accuracy_score(y[te], clf.predict(Xte)))
            fold_means.append(np.mean(mc_acc))
        results[k] = {"ms": round(ms, 3), "acc_mean": float(np.mean(fold_means)),
                      "acc_std": float(np.std(fold_means))}
        print(f"  k={k:3d} ({ms:6.2f} ms): acc = {results[k]['acc_mean']:.3f} "
              f"+/- {results[k]['acc_std']:.3f}")
    return results


def plot(results):
    ms = [results[k]["ms"] for k in K_FRAMES]
    mean = np.array([results[k]["acc_mean"] for k in K_FRAMES])
    std = np.array([results[k]["acc_std"] for k in K_FRAMES])

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    ax.fill_between(ms, mean - std, mean + std, color=C_BAND, alpha=0.5, lw=0)
    ax.plot(ms, mean, color=C_LINE, lw=2, marker="o", ms=6, zorder=5)
    ax.axhline(1 / 7, color=C_REF, lw=1.2, ls=":", zorder=1)
    ax.text(ms[-1], 1 / 7 + 0.015, "chance (1/7)", ha="right", fontsize=8, color=C_REF)
    ax.set_xscale("log")
    ax.set_xticks(ms)
    ax.set_xticklabels([f"{m:.2f}" if m < 1 else f"{m:.0f}" for m in ms], fontsize=8)
    ax.set_xlabel("Observation window length (ms, log scale)", fontsize=10, color=INK2)
    ax.set_ylabel("Drone-model accuracy (single window)", fontsize=10, color=INK2)
    ax.set_ylim(0, 1.02)
    ax.set_title("How short can the window be? — single-window LDA accuracy vs. length\n"
                 "leave-one-run-out, all interference pooled, 256-bin PSD",
                 fontsize=11, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for m, a in zip(ms, mean):
        ax.text(m, a + 0.03, f"{a:.2f}", ha="center", fontsize=8, color=INK)
    fig.tight_layout()
    out = RESULTS_DIR / "segment_length_sweep.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out.resolve()}")


def main():
    meta = pd.read_parquet(SPEC_META)
    keep = (meta["seg_clip_ratio"] <= BAD_CLIP).to_numpy()
    meta = meta[keep].reset_index(drop=True)
    specs = np.load(SPECS_NPY)[keep].astype(np.float32)
    print(f"Loaded {specs.shape} spectrograms; sweeping window length "
          f"{K_FRAMES[0]}-{K_FRAMES[-1]} frames ({MS_PER_FRAME:.3f} ms/frame)")

    results = sweep(specs, meta)
    plot(results)
    (RESULTS_DIR / "segment_length_sweep.json").write_text(json.dumps(
        {"ms_per_frame": MS_PER_FRAME, "n_mc": N_MC, "results": results}, indent=2))
    print(f"Wrote {(RESULTS_DIR / 'segment_length_sweep.json').resolve()}")


if __name__ == "__main__":
    main()
