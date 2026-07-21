### Multi-window majority voting: is it better to spend a fixed observation ###
### budget on ONE long window, or on SEVERAL short windows that vote? ###
### For a base window of k frames we split the 128-frame (50 ms) spectrogram ###
### into non-overlapping k-slots, train the LDA on single k-windows (as in   ###
### the length sweep), and at test time draw V non-overlapping slots per      ###
### recording, classify each, and majority-vote. Total observation time =     ###
### V * k * 0.39 ms, so voting curves are directly comparable to the single-  ###
### window sweep on a shared "observation time" x-axis.                       ###

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
SWEEP_JSON = SCRIPT_DIR / ".." / "results" / "segment_length_sweep.json"
RESULTS_DIR = SCRIPT_DIR / ".." / "results"

DRONE_ORDER = ["AIR", "DIS", "INS", "MIN", "MP1", "MP2", "PHA"]
BAD_CLIP = 0.05
TOTAL_FRAMES = 128
MS_PER_FRAME = 50.0 / TOTAL_FRAMES
N_MC = 10
# base window length -> odd vote counts (majority, no ties) fitting in 128 frames
VOTE_PLAN = {8: [1, 3, 5, 7, 9, 11, 13, 15], 16: [1, 3, 5, 7], 32: [1, 3]}

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
LINES = {8: "#2a78d6", 16: "#1baf7a", 32: "#eda100"}
C_REF, C_CHANCE = "#52514e", "#898781"


def slot_features(spec_row, k):
    """All non-overlapping k-slot PSD vectors for one recording, z-scored."""
    n_slots = TOTAL_FRAMES // k
    feats = np.stack([spec_row[:, s * k:(s + 1) * k].mean(axis=1) for s in range(n_slots)])
    mu = feats.mean(axis=1, keepdims=True)
    sd = feats.std(axis=1, keepdims=True) + 1e-6
    return (feats - mu) / sd                       # (n_slots, 256)


def majority(preds):
    """Row-wise majority vote over (N, V) integer predictions."""
    return np.array([np.bincount(row).argmax() for row in preds])


def run_base_window(specs, meta, y, runs, k):
    n_slots = TOTAL_FRAMES // k
    # precompute all slot features per recording once
    all_slots = np.stack([slot_features(specs[i], k) for i in range(len(specs))])  # (N, n_slots, 256)
    curve = []
    for V in VOTE_PLAN[k]:
        fold_means = []
        for fold in runs:
            te = (meta["run_index"] == fold).to_numpy()
            # train on one random slot per training recording
            tr_idx = np.where(~te)[0]
            pick = RNG.integers(0, n_slots, size=len(tr_idx))
            Xtr = all_slots[tr_idx, pick]
            clf = LinearDiscriminantAnalysis().fit(Xtr, y[tr_idx])
            te_idx = np.where(te)[0]
            hard_acc, soft_acc = [], []
            for _ in range(N_MC):
                # V distinct non-overlapping slots per test recording
                chosen = np.stack([RNG.permutation(n_slots)[:V] for _ in te_idx])  # (Nte, V)
                votes = np.empty((len(te_idx), V), dtype=int)
                proba_sum = np.zeros((len(te_idx), len(DRONE_ORDER)))
                for col in range(V):
                    Xte = all_slots[te_idx, chosen[:, col]]
                    votes[:, col] = clf.predict(Xte)
                    proba_sum += clf.predict_proba(Xte)
                hard_acc.append(accuracy_score(y[te_idx], majority(votes)))
                soft_acc.append(accuracy_score(y[te_idx], proba_sum.argmax(axis=1)))
            fold_means.append((np.mean(hard_acc), np.mean(soft_acc)))
        obs_ms = V * k * MS_PER_FRAME
        fm = np.array(fold_means)
        curve.append({"votes": V, "obs_ms": round(obs_ms, 2),
                      "hard_mean": float(fm[:, 0].mean()), "hard_std": float(fm[:, 0].std()),
                      "soft_mean": float(fm[:, 1].mean()), "soft_std": float(fm[:, 1].std())})
        print(f"  base {k * MS_PER_FRAME:5.2f}ms x {V:2d} votes (obs {obs_ms:5.1f}ms): "
              f"hard = {curve[-1]['hard_mean']:.3f}, soft = {curve[-1]['soft_mean']:.3f}")
    return curve


def main():
    meta = pd.read_parquet(SPEC_META)
    keep = (meta["seg_clip_ratio"] <= BAD_CLIP).to_numpy()
    meta = meta[keep].reset_index(drop=True)
    specs = np.load(SPECS_NPY)[keep].astype(np.float32)
    y = np.array([DRONE_ORDER.index(d) for d in meta["drone_id"]])
    runs = sorted(meta["run_index"].unique())
    print(f"Loaded {specs.shape}; multi-window voting for base windows "
          f"{[round(k * MS_PER_FRAME, 2) for k in VOTE_PLAN]} ms")

    results = {}
    for k in VOTE_PLAN:
        results[k] = run_base_window(specs, meta, y, runs, k)

    # single-window reference curve from the length sweep
    sweep = json.loads(SWEEP_JSON.read_text())["results"] if SWEEP_JSON.exists() else {}
    sw_ms = [v["ms"] for v in sweep.values()]
    sw_acc = [v["acc_mean"] for v in sweep.values()]

    fig, ax = plt.subplots(figsize=(9.5, 5.6), facecolor=SURFACE)
    if sw_ms:
        ax.plot(sw_ms, sw_acc, color=C_REF, lw=1.8, ls="--", marker="s", ms=4,
                label="single window (sweep)", zorder=3)
    for k in VOTE_PLAN:
        ms = [c["obs_ms"] for c in results[k]]
        soft = [c["soft_mean"] for c in results[k]]
        hard = [c["hard_mean"] for c in results[k]]
        ax.plot(ms, soft, color=LINES[k], lw=2, marker="o", ms=6,
                label=f"{k * MS_PER_FRAME:.1f} ms x N (soft vote)", zorder=5)
        ax.plot(ms, hard, color=LINES[k], lw=1.2, ls=":", marker="o", ms=3,
                alpha=0.7, zorder=4)
    ax.axhline(1 / 7, color=C_CHANCE, lw=1.2, ls=":", zorder=1)
    ax.text(45, 1 / 7 + 0.015, "chance (1/7)", ha="right", fontsize=8, color=C_CHANCE)
    ax.set_xscale("log")
    ax.set_xlabel("Total observation time (ms, log scale)", fontsize=10, color=INK2)
    ax.set_ylabel("Drone-model accuracy", fontsize=10, color=INK2)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Multi-window voting vs. one long window, at equal observation time\n"
                 "solid = soft vote, dotted = hard vote; leave-one-run-out, 256-bin PSD",
                 fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = RESULTS_DIR / "multiwindow_voting.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out.resolve()}")

    (RESULTS_DIR / "multiwindow_voting.json").write_text(json.dumps(
        {"ms_per_frame": MS_PER_FRAME, "n_mc": N_MC,
         "results": {f"{k * MS_PER_FRAME:.2f}ms": results[k] for k in VOTE_PLAN}}, indent=2))
    print(f"Wrote {(RESULTS_DIR / 'multiwindow_voting.json').resolve()}")


if __name__ == "__main__":
    main()
