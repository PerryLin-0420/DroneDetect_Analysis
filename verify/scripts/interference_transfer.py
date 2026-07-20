### Robustness check 1: interference-transfer matrix ###
### Trains LDA on segments from one interference condition and tests on each ###
### other condition (4x4 matrix). Diagonal cells use leave-one-run-out inside ###
### the same condition so they are held-out too, not resubstitution. A large  ###
### off-diagonal drop means the classifier leans on the ambient-spectrum      ###
### context rather than the drone signal itself.                              ###
### Note: PHA has no FY recordings under clean/bluetooth, so class            ###
### composition varies slightly across cells; accuracy is still comparable.   ###

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
FEATURES_PARQUET = SCRIPT_DIR / ".." / ".." / "embedding" / "results" / "psd_features.parquet"
RESULTS_DIR = SCRIPT_DIR / ".." / "results"

CONDITIONS = ["clean", "bluetooth", "wifi", "bluetooth_wifi"]
BAD_CLIP_THRESHOLD = 0.05

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load():
    df = pd.read_parquet(FEATURES_PARQUET)
    df = df[df["seg_clip_ratio"] <= BAD_CLIP_THRESHOLD].reset_index(drop=True)
    X = np.stack(df["psd_db"].to_numpy())
    return df, X


def diagonal_acc(df, X, cond):
    """Held-out accuracy inside one condition: leave-one-run-out."""
    mask = (df["interference"] == cond).to_numpy()
    sub, Xs = df[mask], X[mask]
    accs = []
    for fold in sorted(sub["run_index"].unique()):
        test = (sub["run_index"] == fold).to_numpy()
        m = LinearDiscriminantAnalysis()
        m.fit(Xs[~test], sub["drone_id"].to_numpy()[~test])
        accs.append(accuracy_score(sub["drone_id"].to_numpy()[test], m.predict(Xs[test])))
    return float(np.mean(accs))


def transfer_acc(df, X, train_cond, test_cond):
    tr = (df["interference"] == train_cond).to_numpy()
    te = (df["interference"] == test_cond).to_numpy()
    m = LinearDiscriminantAnalysis()
    m.fit(X[tr], df["drone_id"].to_numpy()[tr])
    return float(accuracy_score(df["drone_id"].to_numpy()[te], m.predict(X[te])))


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df, X = load()
    print(f"{len(df)} segments after clip filter")

    acc = np.zeros((len(CONDITIONS), len(CONDITIONS)))
    for i, tr in enumerate(CONDITIONS):
        for j, te in enumerate(CONDITIONS):
            acc[i, j] = diagonal_acc(df, X, tr) if tr == te else transfer_acc(df, X, tr, te)
            print(f"train={tr:15s} test={te:15s} acc={acc[i, j]:.3f}")

    fig, ax = plt.subplots(figsize=(7.2, 6), facecolor=SURFACE)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", [SURFACE] + SEQ_RAMP)
    ax.imshow(acc, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(CONDITIONS)), CONDITIONS, fontsize=9, color=INK2)
    ax.set_yticks(range(len(CONDITIONS)), CONDITIONS, fontsize=9, color=INK2)
    ax.set_xlabel("Test condition", fontsize=10, color=INK2)
    ax.set_ylabel("Train condition", fontsize=10, color=INK2)
    ax.set_title("LDA drone-model accuracy under interference transfer\n"
                 "(diagonal = leave-one-run-out within condition)",
                 fontsize=11, color=INK, loc="left")
    for i in range(len(CONDITIONS)):
        for j in range(len(CONDITIONS)):
            v = acc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11,
                    color=SURFACE if v > 0.55 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    out = RESULTS_DIR / "interference_transfer.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out.resolve()}")

    (RESULTS_DIR / "interference_transfer.json").write_text(json.dumps(
        {"conditions": CONDITIONS, "accuracy": np.round(acc, 4).tolist()}, indent=2))
    print(f"Wrote {(RESULTS_DIR / 'interference_transfer.json').resolve()}")


if __name__ == "__main__":
    main()
