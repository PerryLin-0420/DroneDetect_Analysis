### Model comparison: do LDA / XGBoost / CNN make DIFFERENT mistakes? ###
### Aligns the three per-segment prediction sets on (relative_path,         ###
### segment_index), then reports pairwise agreement, McNemar's test (are the ###
### error sets significantly different?), exclusive-correct counts, and a    ###
### 3-model majority-vote ensemble. A significant McNemar result + ensemble  ###
### gain means the models learned complementary cues, not the same ones.     ###

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_PRED = SCRIPT_DIR / ".." / ".." / "embedding" / "results" / "baseline_predictions.parquet"
CNN_PRED = SCRIPT_DIR / ".." / ".." / "CNN" / "results" / "cnn_predictions.parquet"
RESULTS_DIR = SCRIPT_DIR / ".." / "results"

KEY = ["relative_path", "segment_index"]


def mcnemar(correct_a, correct_b):
    """McNemar on paired correctness. Returns (b01, b10, chi2_cc, p)."""
    b01 = int(np.sum(~correct_a & correct_b))   # a wrong, b right
    b10 = int(np.sum(correct_a & ~correct_b))   # a right, b wrong
    n = b01 + b10
    if n == 0:
        return b01, b10, 0.0, 1.0
    stat = (abs(b01 - b10) - 1) ** 2 / n        # continuity-corrected
    return b01, b10, float(stat), float(chi2.sf(stat, 1))


def main():
    base = pd.read_parquet(BASELINE_PRED)
    cnn = pd.read_parquet(CNN_PRED)[KEY + ["pred_cnn"]]
    df = base.merge(cnn, on=KEY, how="inner")
    print(f"Aligned {len(df)} segments across all three models "
          f"(base {len(base)}, cnn {len(cnn)})")

    truth = df["drone_id"].to_numpy()
    models = {"LDA": "pred_lda", "XGBoost": "pred_xgboost", "CNN": "pred_cnn"}
    correct = {name: (df[col].to_numpy() == truth) for name, col in models.items()}

    print("\n=== individual accuracy (on aligned set) ===")
    for name in models:
        print(f"  {name:8s} {correct[name].mean():.4f}")

    print("\n=== pairwise agreement & McNemar ===")
    names = list(models)
    mc_out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            agree = np.mean(df[models[a]].to_numpy() == df[models[b]].to_numpy())
            b01, b10, stat, p = mcnemar(correct[a], correct[b])
            print(f"  {a} vs {b}: agreement {agree:.4f} | "
                  f"{b}-only-right {b01}, {a}-only-right {b10} | "
                  f"McNemar chi2 {stat:.1f}, p {p:.2e}")
            mc_out[f"{a}_vs_{b}"] = {"agreement": round(float(agree), 4),
                                    f"{b}_only_right": b01, f"{a}_only_right": b10,
                                    "mcnemar_chi2": round(stat, 2), "p_value": p}

    # exclusive correctness: how many segments only ONE model gets right
    stack = np.vstack([correct[n] for n in names])   # (3, N)
    n_right = stack.sum(axis=0)
    print("\n=== how many of the 3 models are right per segment ===")
    for k in range(4):
        print(f"  exactly {k} correct: {int(np.sum(n_right == k))} "
              f"({np.mean(n_right == k) * 100:.1f}%)")

    # 3-model majority vote (ties -> LDA, the strongest single model)
    preds = np.vstack([df[models[n]].to_numpy() for n in names]).T  # (N, 3)
    ensemble = []
    for row in preds:
        vals, counts = np.unique(row, return_counts=True)
        top = vals[counts == counts.max()]
        ensemble.append(row[0] if len(top) > 1 else top[0])  # tie -> LDA (col 0)
    ens_acc = np.mean(np.array(ensemble) == truth)
    print(f"\n=== 3-model majority-vote ensemble ===")
    print(f"  ensemble segment acc = {ens_acc:.4f}  "
          f"(best single = {max(c.mean() for c in correct.values()):.4f})")

    out = {"aligned_segments": len(df),
           "individual_acc": {n: round(float(correct[n].mean()), 4) for n in names},
           "pairwise": mc_out,
           "n_models_correct": {str(k): int(np.sum(n_right == k)) for k in range(4)},
           "ensemble_acc": round(float(ens_acc), 4)}
    (RESULTS_DIR / "model_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {(RESULTS_DIR / 'model_comparison.json').resolve()}")


if __name__ == "__main__":
    main()
