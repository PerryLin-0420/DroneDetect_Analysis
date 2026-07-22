### Resolution sweep summary: the key finding is that raising frequency ###
### resolution helps or hurts depending on HOW it is used. ###
### Left  : baseline accuracy vs. bins -- 2D-spectrogram CNN (per-frame, noisy) ###
###         DROPS with resolution, while LDA on the time-averaged PSD RISES.    ###
### Right : interference-transfer off-diagonal accuracy (absolute) vs. bins --  ###
###         all CNN resolutions sit far below LDA, and the CNN's shrinking      ###
###         'drop' is an artifact of a low in-distribution ceiling.             ###

import json
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
R = SCRIPT_DIR / ".." / ".." / "CNN" / "results"
V = SCRIPT_DIR / ".." / "results"
BINS = [256, 512, 1024]

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
C_CNN, C_PSD, C_LDA = "#e34948", "#2a78d6", "#008300"


def sfx(b):
    return "" if b == 256 else f"_{b}"


def main():
    cnn_base = [json.loads((R / f"cnn_metrics{sfx(b)}.json").read_text())["segment_acc"] for b in BINS]
    psd_full = [json.loads((V / f"segment_length_sweep{sfx(b)}.json").read_text())["results"]["128"]["acc_mean"] for b in BINS]
    cnn_off = [json.loads((V / f"cnn_vs_lda_interference_transfer{sfx(b)}.json").read_text())["cnn_diag_offdiag_drop"][1] for b in BINS]
    lda_tr = json.loads((V / "cnn_vs_lda_interference_transfer.json").read_text())
    lda_off = lda_tr["lda_diag_offdiag_drop"][1]
    lda_base = 0.972

    x = list(range(len(BINS)))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=SURFACE)

    # left: baseline accuracy — the divergence
    axL.plot(x, cnn_base, color=C_CNN, lw=2.2, marker="o", ms=7, label="2D-spectrogram CNN (per-frame)")
    axL.plot(x, psd_full, color=C_PSD, lw=2.2, marker="s", ms=7, label="LDA on time-averaged PSD")
    axL.axhline(lda_base, color=C_LDA, lw=1.4, ls="--", label="native LDA baseline (0.972)")
    for xi, (c, p) in enumerate(zip(cnn_base, psd_full)):
        axL.text(xi, c - 0.012, f"{c:.3f}", ha="center", va="top", fontsize=8, color=C_CNN)
        axL.text(xi, p + 0.012, f"{p:.3f}", ha="center", va="bottom", fontsize=8, color=C_PSD)
    axL.set_xticks(x, [str(b) for b in BINS], fontsize=9, color=INK2)
    axL.set_xlabel("Frequency bins", fontsize=10, color=INK2)
    axL.set_ylabel("Drone-model accuracy", fontsize=10, color=INK2)
    axL.set_ylim(0.84, 0.99)
    axL.set_title("Higher resolution: hurts the CNN, helps the PSD",
                  fontsize=11, color=INK, loc="left")
    axL.legend(frameon=False, fontsize=8.5, loc="center left")

    # right: interference-transfer off-diagonal (absolute), CNN resolutions vs LDA
    axR.plot(x, cnn_off, color=C_CNN, lw=2.2, marker="o", ms=7, label="CNN off-diagonal (cross-interference)")
    axR.axhline(lda_off, color=C_LDA, lw=1.4, ls="--", label=f"LDA off-diagonal ({lda_off:.2f})")
    for xi, c in enumerate(cnn_off):
        axR.text(xi, c - 0.012, f"{c:.3f}", ha="center", va="top", fontsize=8, color=C_CNN)
    axR.set_xticks(x, [str(b) for b in BINS], fontsize=9, color=INK2)
    axR.set_xlabel("Frequency bins", fontsize=10, color=INK2)
    axR.set_ylabel("Cross-interference accuracy (off-diagonal)", fontsize=10, color=INK2)
    axR.set_ylim(0.6, 0.88)
    axR.set_title("Abs. robustness: every CNN resolution trails LDA",
                  fontsize=11, color=INK, loc="left")
    axR.legend(frameon=False, fontsize=8.5, loc="upper right")

    for ax in (axL, axR):
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Resolution sweep — it is time-averaging, not frequency resolution or model depth, that wins",
                 fontsize=12, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = V / "resolution_summary.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
