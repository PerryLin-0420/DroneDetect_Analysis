### Gain-perturbation stress test: is the normalized PSD truly gain-invariant? ###
### An IQ gain of g scales linear power by g^2, i.e. adds a constant offset in ###
### the dB (spectrogram) domain. We apply a test-time offset of Delta dB to    ###
### every test segment and measure accuracy for two feature pipelines:         ###
###   - normalized: per-vector z-score across frequency (removes the offset)   ###
###   - raw: log-power with no normalization (keeps the offset)                 ###
### The LDA is trained once at 0 dB. A flat curve for the normalized pipeline   ###
### and a collapsing curve for the raw one confirm normalization == invariance. ###

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
SPECS_NPY = SCRIPT_DIR / ".." / ".." / "CNN" / "results" / "spectrograms.npy"
SPEC_META = SCRIPT_DIR / ".." / ".." / "CNN" / "results" / "spectrogram_meta.parquet"
RESULTS_DIR = SCRIPT_DIR / ".." / "results"

DRONE_ORDER = ["AIR", "DIS", "INS", "MIN", "MP1", "MP2", "PHA"]
BAD_CLIP = 0.05
DELTAS_DB = [-20, -15, -10, -5, 0, 5, 10, 15, 20]

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
C_NORM, C_RAW, C_CHANCE = "#1baf7a", "#e34948", "#898781"


def collapse_time(specs):
    """50 ms spectrogram -> full-window PSD vector (mean over time), (N, 256) dB."""
    return specs.mean(axis=2)


def zscore(feats):
    mu = feats.mean(axis=1, keepdims=True)
    sd = feats.std(axis=1, keepdims=True) + 1e-6
    return (feats - mu) / sd


def evaluate(train_feat, y, tr, te, delta_db, normalize):
    Xtr = zscore(train_feat[tr]) if normalize else train_feat[tr]
    clf = LinearDiscriminantAnalysis().fit(Xtr, y[tr])
    # apply the gain offset (dB domain) to the TEST features only
    Xte = train_feat[te] + delta_db
    Xte = zscore(Xte) if normalize else Xte
    return accuracy_score(y[te], clf.predict(Xte))


def main():
    meta = pd.read_parquet(SPEC_META)
    keep = (meta["seg_clip_ratio"] <= BAD_CLIP).to_numpy()
    meta = meta[keep].reset_index(drop=True)
    specs = np.load(SPECS_NPY)[keep].astype(np.float32)
    feat = collapse_time(specs)                       # (N, 256) dB PSD
    y = np.array([DRONE_ORDER.index(d) for d in meta["drone_id"]])
    runs = sorted(meta["run_index"].unique())
    print(f"Loaded {feat.shape} PSD vectors; sweeping test-time gain {DELTAS_DB} dB")

    curves = {"normalized": [], "raw": []}
    for mode, normalize in [("normalized", True), ("raw", False)]:
        for d in DELTAS_DB:
            fold_acc = []
            for fold in runs:
                te = (meta["run_index"] == fold).to_numpy()
                fold_acc.append(evaluate(feat, y, ~te, te, d, normalize))
            curves[mode].append(float(np.mean(fold_acc)))
        print(f"  {mode:11s}: " + " ".join(f"{a:.2f}" for a in curves[mode]))

    fig, ax = plt.subplots(figsize=(8.5, 5), facecolor=SURFACE)
    ax.plot(DELTAS_DB, curves["normalized"], color=C_NORM, lw=2.2, marker="o", ms=6,
            label="normalized PSD (per-vector z-score)", zorder=5)
    ax.plot(DELTAS_DB, curves["raw"], color=C_RAW, lw=2.2, marker="s", ms=6,
            label="raw log-power (no normalization)", zorder=5)
    ax.axhline(1 / 7, color=C_CHANCE, lw=1.2, ls=":", zorder=1)
    ax.text(20, 1 / 7 + 0.02, "chance (1/7)", ha="right", fontsize=8, color=C_CHANCE)
    ax.set_xlabel("Test-time gain offset (dB)", fontsize=10, color=INK2)
    ax.set_ylabel("Drone-model accuracy", fontsize=10, color=INK2)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(DELTAS_DB)
    ax.set_title("Gain-perturbation stress test — normalized PSD is gain-invariant\n"
                 "LDA trained at 0 dB, leave-one-run-out",
                 fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower center")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = RESULTS_DIR / "gain_perturbation.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out.resolve()}")

    (RESULTS_DIR / "gain_perturbation.json").write_text(json.dumps(
        {"deltas_db": DELTAS_DB, "normalized": curves["normalized"], "raw": curves["raw"]},
        indent=2))
    print(f"Wrote {(RESULTS_DIR / 'gain_perturbation.json').resolve()}")


if __name__ == "__main__":
    main()
