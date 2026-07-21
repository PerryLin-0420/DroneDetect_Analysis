### Session-leakage verification: does the representation encode the drone ###
### model, or a per-recording/session fingerprint? ###
### Two probes on two feature sets (CNN 128-d embedding, PSD 1024-d):        ###
###   - linear logistic-regression probe for drone_id (main-task reference), ###
###     run_index (session/run leakage), interference, flight_mode           ###
###   - GroupKFold by recording so adjacent segments never straddle folds    ###
### Plus linear CKA between CNN embedding and PSD features (representation    ###
### similarity). Probing both feature sets separates "leakage inherent to    ###
### the signal" from "leakage introduced by the CNN".                        ###

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
CNN_EMB = SCRIPT_DIR / ".." / ".." / "CNN" / "results" / "cnn_embeddings.npy"
CNN_PRED = SCRIPT_DIR / ".." / ".." / "CNN" / "results" / "cnn_predictions.parquet"
PSD_FEATURES = SCRIPT_DIR / ".." / ".." / "embedding" / "results" / "psd_features.parquet"
RESULTS_DIR = SCRIPT_DIR / ".." / "results"

TARGETS = ["drone_id", "run_index", "interference", "flight_mode"]
KEY = ["relative_path", "segment_index"]

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
C_CNN, C_PSD, C_CHANCE = "#2a78d6", "#1baf7a", "#e34948"


def load_aligned():
    """CNN embedding + its meta, joined to matching PSD feature vectors."""
    emb = np.load(CNN_EMB)                      # (N, 128), order == cnn_pred meta
    meta = pd.read_parquet(CNN_PRED)
    assert len(emb) == len(meta), f"emb {len(emb)} != meta {len(meta)}"

    psd = pd.read_parquet(PSD_FEATURES)[KEY + ["psd_db"]]
    meta = meta.reset_index(drop=True)
    meta["_row"] = np.arange(len(meta))
    merged = meta.merge(psd, on=KEY, how="inner").sort_values("_row")
    emb = emb[merged["_row"].to_numpy()]
    psd_mat = np.stack(merged["psd_db"].to_numpy())
    print(f"Aligned {len(merged)} segments: CNN emb {emb.shape}, PSD {psd_mat.shape}")
    return merged.reset_index(drop=True), emb, psd_mat


def probe(X, y, groups, tag):
    """Linear probe accuracy under GroupKFold (recording-level)."""
    gkf = GroupKFold(n_splits=5)
    accs = []
    for tr, te in gkf.split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(scaler.transform(X[tr]), y[tr])
        accs.append(clf.score(scaler.transform(X[te]), y[te]))
    return float(np.mean(accs)), float(np.std(accs))


# The CNN embedding is generated per leave-one-run-out fold (5 distinct CNNs),
# so a run_index probe on it just recovers "which fold's model emitted this
# vector" -- an artifact, not a signal fingerprint. The valid run-leakage test
# is the PSD probe, which touches no model.
ARTIFACT = {("CNN embedding", "run_index")}


def linear_cka(X, Y):
    """Feature-space linear CKA (no n x n gram matrix)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    nx = np.linalg.norm(Xc.T @ Xc, "fro")
    ny = np.linalg.norm(Yc.T @ Yc, "fro")
    return float(cross / (nx * ny))


def main():
    meta, emb, psd = load_aligned()
    groups = meta["relative_path"].to_numpy()
    feature_sets = {"CNN embedding": emb, "PSD features": psd}

    chance = {t: float(meta[t].value_counts(normalize=True).max()) for t in TARGETS}
    print("\nMajority-class baseline (chance):",
          {t: round(c, 3) for t, c in chance.items()})

    results = {}
    print("\n=== linear probe accuracy (GroupKFold by recording) ===")
    for fs_name, X in feature_sets.items():
        results[fs_name] = {}
        for t in TARGETS:
            y = meta[t].astype(str).to_numpy()
            acc, sd = probe(X, y, groups, f"{fs_name}/{t}")
            is_artifact = (fs_name, t) in ARTIFACT
            results[fs_name][t] = {"acc": round(acc, 4), "std": round(sd, 4),
                                   "chance": round(chance[t], 4),
                                   "artifact": is_artifact}
            flag = "  [ARTIFACT: per-fold embedding, not valid]" if is_artifact else ""
            print(f"  {fs_name:14s} {t:13s} acc = {acc:.3f} +/- {sd:.3f}  "
                  f"(chance {chance[t]:.3f}){flag}")

    cka = linear_cka(emb, psd)
    print(f"\n=== linear CKA (CNN embedding vs PSD features) = {cka:.3f} ===")

    # grouped bar chart: probe accuracy per target, CNN vs PSD, with chance line
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    x = np.arange(len(TARGETS))
    w = 0.36
    cnn_acc = [results["CNN embedding"][t]["acc"] for t in TARGETS]
    psd_acc = [results["PSD features"][t]["acc"] for t in TARGETS]
    # hatch + faded any artifact bar so it can't be read as a real result
    cnn_hatch = ["//" if (("CNN embedding", t) in ARTIFACT) else "" for t in TARGETS]
    cnn_alpha = [0.35 if (("CNN embedding", t) in ARTIFACT) else 1.0 for t in TARGETS]
    bars = ax.bar(x - w / 2, cnn_acc, w, color=C_CNN, label="CNN embedding (128-d)")
    for b, h, a in zip(bars, cnn_hatch, cnn_alpha):
        b.set_hatch(h)
        b.set_alpha(a)
    ax.bar(x + w / 2, psd_acc, w, color=C_PSD, label="PSD features (1024-d)")
    for xi, t in enumerate(TARGETS):
        ax.plot([xi - w, xi + w], [chance[t], chance[t]], color=C_CHANCE,
                lw=2, zorder=5, label="chance" if xi == 0 else None)
    ax.set_xticks(x, [f"{t}\n(chance {chance[t]:.2f})" for t in TARGETS],
                  fontsize=9, color=INK2)
    ax.set_ylabel("Linear probe accuracy", fontsize=10, color=INK2)
    ax.set_ylim(0, 1.05)
    ax.set_title("What the representations encode — drone_id is the task; "
                 "run_index measures session leakage", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for xi, (c, p) in enumerate(zip(cnn_acc, psd_acc)):
        ax.text(xi - w / 2, c + 0.02, f"{c:.2f}", ha="center", fontsize=8, color=INK)
        ax.text(xi + w / 2, p + 0.02, f"{p:.2f}", ha="center", fontsize=8, color=INK)
        if ("CNN embedding", TARGETS[xi]) in ARTIFACT:
            ax.text(xi - w / 2, 0.5, "artifact\n(per-fold\nembedding)", ha="center",
                    va="center", fontsize=7.5, color=INK, rotation=0, style="italic")
    fig.tight_layout()
    out = RESULTS_DIR / "session_leakage_probe.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out.resolve()}")

    (RESULTS_DIR / "session_leakage.json").write_text(json.dumps(
        {"probe": results, "cka_cnn_vs_psd": round(cka, 4)}, indent=2))
    print(f"Wrote {(RESULTS_DIR / 'session_leakage.json').resolve()}")


if __name__ == "__main__":
    main()
