### Small 2D CNN on pooled log-spectrograms, leave-one-run-out CV ###
### Input: (256F x 128T) float16 spectrograms + meta parquet from            ###
### extract_spectrograms.py. Per-segment z-score at load time removes the    ###
### gain confound (log makes gain an additive constant). Light augmentation  ###
### (random gain offset is a no-op after z-score, so we use time roll +      ###
### noise) discourages session shortcuts. CPU-friendly: ~200k params.        ###
### Outputs per-fold metrics, pooled confusion matrix, per-segment           ###
### predictions, and penultimate-layer embeddings for the verify/ stage.     ###

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix

sys.stdout.reconfigure(line_buffering=True)
torch.manual_seed(0)
np.random.seed(0)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / ".." / "results"
MODELS_DIR = SCRIPT_DIR / ".." / "models"  # saved per-fold model weights
# frequency bins: optional CLI arg (256 default); matches extract_spectrograms.py output
FREQ_BINS = int(sys.argv[1]) if len(sys.argv) > 1 else 256
SUFFIX = "" if FREQ_BINS == 256 else f"_{FREQ_BINS}"  # keep 256 filenames unchanged
SPECS_NPY = RESULTS_DIR / f"spectrograms{SUFFIX}.npy"
META_PARQUET = RESULTS_DIR / f"spectrogram_meta{SUFFIX}.parquet"

DRONE_ORDER = ["AIR", "DIS", "INS", "MIN", "MP1", "MP2", "PHA"]
BAD_CLIP_THRESHOLD = 0.05
EPOCHS = 12
BATCH = 128
LR = 3e-4

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


class SmallCNN(nn.Module):
    """4 conv blocks -> GAP -> linear head. ~200k params, CPU-trainable."""

    def __init__(self, n_classes: int):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.features = nn.Sequential(
            block(1, 16), block(16, 32), block(32, 64), block(64, 128))
        self.head = nn.Linear(128, n_classes)

    def forward(self, x, return_embedding=False):
        z = self.features(x).mean(dim=(2, 3))  # global average pool -> (B, 128)
        if return_embedding:
            return z
        return self.head(z)


def load_data():
    specs = np.load(SPECS_NPY)                       # (N, 256, 128) float16
    meta = pd.read_parquet(META_PARQUET)
    keep = (meta["seg_clip_ratio"] <= BAD_CLIP_THRESHOLD).to_numpy()
    specs, meta = specs[keep], meta[keep].reset_index(drop=True)

    # per-segment z-score in float32: removes gain (additive in log domain)
    X = specs.astype(np.float32)
    mu = X.mean(axis=(1, 2), keepdims=True)
    sd = X.std(axis=(1, 2), keepdims=True) + 1e-6
    X = (X - mu) / sd
    y = np.array([DRONE_ORDER.index(d) for d in meta["drone_id"]])
    print(f"Loaded {X.shape} segments after clip filter")
    return X, y, meta


def augment(xb: torch.Tensor) -> torch.Tensor:
    """Random circular time roll + light Gaussian noise (session-shortcut breaker)."""
    shift = int(torch.randint(0, xb.shape[-1], (1,)))
    xb = torch.roll(xb, shifts=shift, dims=-1)
    return xb + 0.05 * torch.randn_like(xb)


def train_fold(X, y, meta, fold):
    test_mask = (meta["run_index"] == fold).to_numpy()
    Xtr = torch.from_numpy(X[~test_mask]).unsqueeze(1)
    ytr = torch.from_numpy(y[~test_mask]).long()
    Xte = torch.from_numpy(X[test_mask]).unsqueeze(1)

    model = SmallCNN(len(DRONE_ORDER))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()

    n = len(Xtr)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        total, t0 = 0.0, time.time()
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = augment(Xtr[idx]), ytr[idx]
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"  fold {fold} epoch {epoch + 1}/{EPOCHS}: "
              f"loss {total / n:.4f} ({time.time() - t0:.0f}s)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = MODELS_DIR / f"cnn_fold_run{fold}{SUFFIX}.pt"
    torch.save({"state_dict": model.state_dict(), "drone_order": DRONE_ORDER,
                "held_out_run": int(fold), "epochs": EPOCHS}, ckpt)
    print(f"  fold {fold}: saved {ckpt.name}")

    model.eval()
    preds, embs = [], []
    with torch.no_grad():
        for i in range(0, len(Xte), 256):
            xb = Xte[i:i + 256]
            preds.append(model(xb).argmax(1).numpy())
            embs.append(model(xb, return_embedding=True).numpy())
    return test_mask, np.concatenate(preds), np.concatenate(embs)


def main():
    X, y, meta = load_data()
    pred_by_row = np.full(len(y), -1, dtype=int)
    emb_by_row = np.zeros((len(y), 128), dtype=np.float32)

    fold_acc = []
    for fold in sorted(meta["run_index"].unique()):
        test_mask, pred, emb = train_fold(X, y, meta, fold)
        pred_by_row[test_mask] = pred
        emb_by_row[test_mask] = emb
        acc = accuracy_score(y[test_mask], pred)
        fold_acc.append(acc)
        print(f"fold run={fold}: segment acc = {acc:.3f}\n")

    seg_acc = accuracy_score(y, pred_by_row)
    votes = meta.assign(true=y, pred=pred_by_row).groupby("relative_path").agg(
        true=("true", "first"), pred=("pred", lambda s: s.mode().iloc[0]))
    rec_acc = accuracy_score(votes["true"], votes["pred"])
    print(f"overall: segment acc = {seg_acc:.3f} (+/- {np.std(fold_acc):.3f}), "
          f"recording majority-vote acc = {rec_acc:.3f}")

    # confusion matrix figure
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cm = confusion_matrix(y, pred_by_row, labels=range(len(DRONE_ORDER)), normalize="true")
    fig, ax = plt.subplots(figsize=(6.6, 5.6), facecolor=SURFACE)
    cmap = LinearSegmentedColormap.from_list("seq_blue", [SURFACE] + SEQ_RAMP)
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(DRONE_ORDER)), DRONE_ORDER, fontsize=9, color=INK2)
    ax.set_yticks(range(len(DRONE_ORDER)), DRONE_ORDER, fontsize=9, color=INK2)
    ax.set_xlabel("Predicted", fontsize=9, color=INK2)
    ax.set_ylabel("True", fontsize=9, color=INK2)
    ax.set_title(f"CNN (spectrogram) - segment acc {seg_acc:.3f}, "
                 f"recording acc {rec_acc:.3f}", fontsize=10, color=INK, loc="left")
    for i in range(len(DRONE_ORDER)):
        for j in range(len(DRONE_ORDER)):
            if cm[i, j] >= 0.005:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color=SURFACE if cm[i, j] > 0.55 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"cnn_confusion{SUFFIX}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    (RESULTS_DIR / f"cnn_metrics{SUFFIX}.json").write_text(json.dumps({
        "fold_acc": [round(a, 4) for a in fold_acc],
        "segment_acc": round(seg_acc, 4),
        "recording_acc": round(rec_acc, 4)}, indent=2))

    # per-segment predictions + embeddings for the verify/ comparison stage
    out = meta[["relative_path", "drone_id", "interference", "flight_mode",
                "run_index", "segment_index"]].copy()
    out["pred_cnn"] = [DRONE_ORDER[p] for p in pred_by_row]
    out.to_parquet(RESULTS_DIR / f"cnn_predictions{SUFFIX}.parquet")
    np.save(RESULTS_DIR / f"cnn_embeddings{SUFFIX}.npy", emb_by_row)
    print(f"Wrote confusion png, metrics json, predictions parquet, embeddings npy "
          f"to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
