### Stage-2 baseline: how separable are the 7 drone models from PSD shape? ###
### Loads the per-segment PSD features, runs leave-one-run-out CV (all       ###
### segments of a recording stay on one side; run_index 0-4 -> 5 folds) with ###
### two models: LDA (linear separability floor) and XGBoost (non-linear      ###
### reference). Reports per-fold accuracy, a pooled confusion matrix, and a  ###
### recording-level majority-vote accuracy. Results land in ../results.      ###

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_PARQUET = SCRIPT_DIR / ".." / "results" / "psd_features.parquet"
RESULTS_DIR = SCRIPT_DIR / ".." / "results"

DRONE_ORDER = ["AIR", "DIS", "INS", "MIN", "MP1", "MP2", "PHA"]
BAD_CLIP_THRESHOLD = 0.05  # exclude segments from grossly saturated recordings

# chart chrome (same validated palette as the EDA plots)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load_features():
    df = pd.read_parquet(FEATURES_PARQUET)
    n_all = len(df)
    df = df[df["seg_clip_ratio"] <= BAD_CLIP_THRESHOLD].reset_index(drop=True)
    print(f"Loaded {n_all} segments; kept {len(df)} after clip filter "
          f"(seg_clip_ratio <= {BAD_CLIP_THRESHOLD})")
    X = np.stack(df["psd_db"].to_numpy())
    y = df["drone_id"].to_numpy()
    return df, X, y


def leave_one_run_out(df, X, y, model_factory, tag):
    """5 folds by run_index; segments of one recording never straddle folds."""
    seg_true, seg_pred, fold_acc = [], [], []
    rec_true, rec_pred = [], []
    pred_by_row = np.empty(len(df), dtype=object)  # predictions in original row order

    for fold in sorted(df["run_index"].unique()):
        test_mask = (df["run_index"] == fold).to_numpy()
        model = model_factory()
        model.fit(X[~test_mask], y[~test_mask])
        pred = model.predict(X[test_mask])
        pred_by_row[test_mask] = pred

        acc = accuracy_score(y[test_mask], pred)
        fold_acc.append(acc)
        seg_true.extend(y[test_mask])
        seg_pred.extend(pred)

        # recording-level majority vote over that recording's segments
        test_df = df.loc[test_mask, ["relative_path", "drone_id"]].copy()
        test_df["pred"] = pred
        votes = test_df.groupby("relative_path").agg(
            true=("drone_id", "first"),
            pred=("pred", lambda s: s.mode().iloc[0]),
        )
        rec_true.extend(votes["true"])
        rec_pred.extend(votes["pred"])
        print(f"  [{tag}] fold run={fold}: segment acc = {acc:.3f}")

    seg_acc = accuracy_score(seg_true, seg_pred)
    rec_acc = accuracy_score(rec_true, rec_pred)
    print(f"  [{tag}] overall: segment acc = {seg_acc:.3f} "
          f"(+/- {np.std(fold_acc):.3f}), recording majority-vote acc = {rec_acc:.3f}")
    cm = confusion_matrix(seg_true, seg_pred, labels=DRONE_ORDER, normalize="true")
    return {"tag": tag, "fold_acc": fold_acc, "segment_acc": seg_acc,
            "recording_acc": rec_acc, "confusion": cm, "pred_by_row": pred_by_row}


def plot_confusion(results):
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 5.4),
                             facecolor=SURFACE)
    if len(results) == 1:
        axes = [axes]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", [SURFACE] + SEQ_RAMP)

    for ax, res in zip(axes, results):
        cm = res["confusion"]
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(DRONE_ORDER)), DRONE_ORDER, fontsize=9, color=INK2)
        ax.set_yticks(range(len(DRONE_ORDER)), DRONE_ORDER, fontsize=9, color=INK2)
        ax.set_xlabel("Predicted", fontsize=9, color=INK2)
        ax.set_ylabel("True", fontsize=9, color=INK2)
        ax.set_title(f"{res['tag']} - segment acc {res['segment_acc']:.3f}, "
                     f"recording acc {res['recording_acc']:.3f}",
                     fontsize=10, color=INK, loc="left")
        for i in range(len(DRONE_ORDER)):
            for j in range(len(DRONE_ORDER)):
                v = cm[i, j]
                if v >= 0.005:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color=SURFACE if v > 0.55 else INK)
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Leave-one-run-out confusion matrices (row-normalized)",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = RESULTS_DIR / "baseline_confusion.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out.resolve()}")


def main():
    df, X, y = load_features()
    print(f"X shape = {X.shape}, classes = {sorted(set(y))}\n")

    results = []
    print("=== LDA (linear separability floor) ===")
    results.append(leave_one_run_out(df, X, y, LinearDiscriminantAnalysis, "LDA"))

    print("\n=== XGBoost (non-linear reference) ===")
    classes = sorted(set(y))
    to_int = {c: i for i, c in enumerate(classes)}

    class XGBWrap:
        def __init__(self):
            self.m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                   tree_method="hist", n_jobs=-1, random_state=0)
        def fit(self, X, y):
            self.m.fit(X, np.array([to_int[v] for v in y]))
        def predict(self, X):
            return np.array([classes[i] for i in self.m.predict(X)])

    results.append(leave_one_run_out(df, X, y, XGBWrap, "XGBoost"))

    plot_confusion(results)

    metrics = {r["tag"]: {"fold_acc": [round(a, 4) for a in r["fold_acc"]],
                          "segment_acc": round(r["segment_acc"], 4),
                          "recording_acc": round(r["recording_acc"], 4)}
               for r in results}
    out_json = RESULTS_DIR / "baseline_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {out_json.resolve()}")

    # persist per-segment predictions for the later model-comparison stage (verify/)
    pred_df = df[["relative_path", "drone_id", "interference", "flight_mode",
                  "run_index", "segment_index"]].copy()
    for r in results:
        pred_df[f"pred_{r['tag'].lower()}"] = r["pred_by_row"]
    pred_df.to_parquet(RESULTS_DIR / "baseline_predictions.parquet")
    print(f"Wrote {(RESULTS_DIR / 'baseline_predictions.parquet').resolve()}")


if __name__ == "__main__":
    main()
