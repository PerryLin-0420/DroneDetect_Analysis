### Closing the loop: does the CNN transfer across interference better than ###
### the PSD baseline, as the interference-invariance probe predicted? ###
### Unified protocol for a fair comparison of both representations:          ###
###   - train on condition i, runs 0-3                                       ###
###   - diagonal cell (i,i): test on condition i, run 4 (held-out)           ###
###   - off-diagonal (i,j): test on condition j, all runs                    ###
### CNN uses the pooled spectrograms; LDA uses the PSD features. Reports two ###
### 4x4 matrices and the mean on-diagonal vs off-diagonal drop for each.     ###

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent / ".." / ".." / "CNN" / "scripts"))
from train_cnn import SmallCNN  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

SCRIPT_DIR = Path(__file__).resolve().parent
# frequency bins: optional CLI arg (256 default) selecting which spectrogram set to use
FREQ_BINS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 256
SUFFIX = "" if FREQ_BINS == 256 else f"_{FREQ_BINS}"
SPECS_NPY = SCRIPT_DIR / ".." / ".." / "CNN" / "results" / f"spectrograms{SUFFIX}.npy"
SPEC_META = SCRIPT_DIR / ".." / ".." / "CNN" / "results" / f"spectrogram_meta{SUFFIX}.parquet"
# LDA always uses its strongest feature (the full 1024-bin PSD) as the fixed baseline
PSD_FEATURES = SCRIPT_DIR / ".." / ".." / "embedding" / "results" / "psd_features.parquet"
RESULTS_DIR = SCRIPT_DIR / ".." / "results"
MODELS_DIR = SCRIPT_DIR / ".." / ".." / "CNN" / "models"  # saved model weights

CONDITIONS = ["clean", "bluetooth", "wifi", "bluetooth_wifi"]
DRONE_ORDER = ["AIR", "DIS", "INS", "MIN", "MP1", "MP2", "PHA"]
BAD_CLIP = 0.05
EPOCHS = 10
BATCH = 128
LR = 3e-4

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def zscore(X):
    mu = X.mean(axis=(1, 2), keepdims=True)
    sd = X.std(axis=(1, 2), keepdims=True) + 1e-6
    return (X - mu) / sd


def augment(xb):
    shift = int(torch.randint(0, xb.shape[-1], (1,)))
    return torch.roll(xb, shifts=shift, dims=-1) + 0.05 * torch.randn_like(xb)


def train_cnn(X, y):
    Xt = torch.from_numpy(X).unsqueeze(1)
    yt = torch.from_numpy(y).long()
    model = SmallCNN(len(DRONE_ORDER))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    n = len(Xt)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(augment(Xt[idx])), yt[idx])
            loss.backward()
            opt.step()
    return model


def cnn_predict(model, X):
    model.eval()
    Xt = torch.from_numpy(X).unsqueeze(1)
    out = []
    with torch.no_grad():
        for i in range(0, len(Xt), 256):
            out.append(model(Xt[i:i + 256]).argmax(1).numpy())
    return np.concatenate(out)


def transfer_matrix_cnn(specs, meta):
    y = np.array([DRONE_ORDER.index(d) for d in meta["drone_id"]])
    acc = np.zeros((4, 4))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for i, tr_cond in enumerate(CONDITIONS):
        tr_mask = ((meta["interference"] == tr_cond) & (meta["run_index"] <= 3)).to_numpy()
        t0 = time.time()
        model = train_cnn(specs[tr_mask], y[tr_mask])
        ckpt = MODELS_DIR / f"transfer_{tr_cond}{SUFFIX}.pt"
        torch.save({"state_dict": model.state_dict(), "drone_order": DRONE_ORDER,
                    "train_condition": tr_cond, "epochs": EPOCHS}, ckpt)
        for j, te_cond in enumerate(CONDITIONS):
            if i == j:
                te_mask = ((meta["interference"] == te_cond) & (meta["run_index"] == 4)).to_numpy()
            else:
                te_mask = (meta["interference"] == te_cond).to_numpy()
            acc[i, j] = accuracy_score(y[te_mask], cnn_predict(model, specs[te_mask]))
        print(f"  [CNN] train={tr_cond:15s} done ({time.time() - t0:.0f}s), "
              f"saved {ckpt.name}  row acc {np.round(acc[i], 3)}")
    return acc


def transfer_matrix_lda(psd, meta):
    y = meta["drone_id"].to_numpy()
    acc = np.zeros((4, 4))
    for i, tr_cond in enumerate(CONDITIONS):
        tr_mask = ((meta["interference"] == tr_cond) & (meta["run_index"] <= 3)).to_numpy()
        clf = LinearDiscriminantAnalysis().fit(psd[tr_mask], y[tr_mask])
        for j, te_cond in enumerate(CONDITIONS):
            if i == j:
                te_mask = ((meta["interference"] == te_cond) & (meta["run_index"] == 4)).to_numpy()
            else:
                te_mask = (meta["interference"] == te_cond).to_numpy()
            acc[i, j] = accuracy_score(y[te_mask], clf.predict(psd[te_mask]))
        print(f"  [LDA] train={tr_cond:15s} row acc {np.round(acc[i], 3)}")
    return acc


def diag_offdiag(acc):
    diag = np.mean(np.diag(acc))
    off = np.mean(acc[~np.eye(4, dtype=bool)])
    return float(diag), float(off), float(diag - off)


def plot_matrices(cnn_acc, lda_acc):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", [SURFACE] + SEQ_RAMP)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=SURFACE)
    for ax, acc, name in zip(axes, [lda_acc, cnn_acc], ["LDA (PSD)", "CNN (spectrogram)"]):
        ax.imshow(acc, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(4), CONDITIONS, fontsize=8, color=INK2, rotation=20, ha="right")
        ax.set_yticks(range(4), CONDITIONS, fontsize=8, color=INK2)
        ax.set_xlabel("Test condition", fontsize=9, color=INK2)
        ax.set_ylabel("Train condition", fontsize=9, color=INK2)
        d, o, drop = diag_offdiag(acc)
        ax.set_title(f"{name}\ndiag {d:.2f} / off-diag {o:.2f} / drop {drop:.2f}",
                     fontsize=10, color=INK, loc="left")
        for i in range(4):
            for j in range(4):
                v = acc[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                        color=SURFACE if v > 0.55 else INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Interference transfer, unified protocol (train cond runs 0-3; "
                 "diag=run4 held-out, off-diag=other cond all)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = RESULTS_DIR / f"cnn_vs_lda_interference_transfer{SUFFIX}.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out.resolve()}")


def main():
    smoke = "--smoke" in sys.argv
    meta = pd.read_parquet(SPEC_META)
    keep = (meta["seg_clip_ratio"] <= BAD_CLIP).to_numpy()
    meta = meta[keep].reset_index(drop=True)
    specs = zscore(np.load(SPECS_NPY)[keep].astype(np.float32))
    print(f"Loaded {specs.shape} spectrograms after clip filter")

    if smoke:  # train only the first condition, report its row, then stop
        y = np.array([DRONE_ORDER.index(d) for d in meta["drone_id"]])
        tr = ((meta["interference"] == "clean") & (meta["run_index"] <= 3)).to_numpy()
        t0 = time.time()
        model = train_cnn(specs[tr], y[tr])
        te = ((meta["interference"] == "wifi")).to_numpy()
        a = accuracy_score(y[te], cnn_predict(model, specs[te]))
        print(f"[SMOKE] clean->wifi acc = {a:.3f}, single-model time {time.time() - t0:.0f}s")
        return

    print("\n=== CNN interference-transfer ===")
    cnn_acc = transfer_matrix_cnn(specs, meta)

    psd_df = pd.read_parquet(PSD_FEATURES)
    psd_df = psd_df[psd_df["seg_clip_ratio"] <= BAD_CLIP].reset_index(drop=True)
    # align PSD rows to the spectrogram meta order
    psd_df = meta.merge(psd_df[["relative_path", "segment_index", "psd_db"]],
                        on=["relative_path", "segment_index"], how="left")
    psd = np.stack(psd_df["psd_db"].to_numpy())
    print("\n=== LDA interference-transfer (same protocol) ===")
    lda_acc = transfer_matrix_lda(psd, meta)

    plot_matrices(cnn_acc, lda_acc)
    cnn_d = diag_offdiag(cnn_acc)
    lda_d = diag_offdiag(lda_acc)
    print(f"\nCNN: diag {cnn_d[0]:.3f}, off-diag {cnn_d[1]:.3f}, drop {cnn_d[2]:.3f}")
    print(f"LDA: diag {lda_d[0]:.3f}, off-diag {lda_d[1]:.3f}, drop {lda_d[2]:.3f}")

    (RESULTS_DIR / f"cnn_vs_lda_interference_transfer{SUFFIX}.json").write_text(json.dumps({
        "conditions": CONDITIONS,
        "cnn_matrix": np.round(cnn_acc, 4).tolist(),
        "lda_matrix": np.round(lda_acc, 4).tolist(),
        "cnn_diag_offdiag_drop": [round(v, 4) for v in cnn_d],
        "lda_diag_offdiag_drop": [round(v, 4) for v in lda_d],
    }, indent=2))
    print(f"Wrote {(RESULTS_DIR / f'cnn_vs_lda_interference_transfer{SUFFIX}.json').resolve()}")


if __name__ == "__main__":
    main()
