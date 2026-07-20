### Basic EDA plots over the 390-row summary table ###
### For each derived feature: box plots grouped by drone model, interference ###
### type, and flight mode (one 3-panel figure per feature), plus a compact    ###
### overview grid. All paths are relative to this script's location.          ###

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_PARQUET = SCRIPT_DIR / ".." / ".." / "Summary_duckdb" / "summary.parquet"
RESULTS_DIR = SCRIPT_DIR / ".." / "results"

# --- palette / chart chrome (validated reference palette, light mode) ---
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
               "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# fixed category orders -> fixed hue assignment (color follows the entity)
DRONE_ORDER = ["AIR", "DIS", "INS", "MIN", "MP1", "MP2", "PHA"]
INTERFERENCE_ORDER = ["clean", "bluetooth", "wifi", "bluetooth_wifi"]
FLIGHT_MODE_ORDER = ["ON", "HO", "FY"]

GROUP_DIMS = [
    ("drone_id", "Drone model", DRONE_ORDER),
    ("interference", "Interference", INTERFERENCE_ORDER),
    ("flight_mode", "Flight mode", FLIGHT_MODE_ORDER),
]

# (column, axis label, log-scale flag)
FEATURES = [
    ("avg_power_db", "Average power (dBFS)", False),
    ("rms_amplitude", "RMS amplitude", False),
    ("peak_power", "Peak power (I2+Q2)", False),
    ("papr", "PAPR (peak / avg power)", True),
    ("std_I", "Std of I", True),
    ("std_Q", "Std of Q", True),
    ("iq_correlation", "I/Q correlation", False),
    ("iq_imbalance_db", "I/Q imbalance (dB)", False),
    ("dc_offset_mag", "DC offset magnitude", True),
    ("zero_ratio", "Zero-sample ratio", True),
    ("clip_ratio", "Clipped-sample ratio (|I| or |Q| >= 0.99)", False),
    ("iqr_I", "IQR of I", True),
    ("max_I", "Max of I", False),
]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "text.color": INK_PRIMARY,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.edgecolor": BASELINE,
    "axes.grid": True,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})


def load_summary():
    con = duckdb.connect()
    df = con.execute(
        f"SELECT * FROM read_parquet('{SUMMARY_PARQUET.as_posix()}')"
    ).df()
    con.close()
    return df


def styled_boxplot(ax, df, group_col, order, value_col, log_scale):
    """Draw one boxplot group on ax with fixed per-category colors."""
    cats = [c for c in order if c in df[group_col].unique()]
    data = [df.loc[df[group_col] == c, value_col].dropna().values for c in cats]

    bp = ax.boxplot(
        data,
        tick_labels=cats,
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, markerfacecolor="none",
                        markeredgecolor=INK_MUTED, markeredgewidth=0.7),
        medianprops=dict(color=INK_PRIMARY, linewidth=1.4),
        whiskerprops=dict(color=INK_MUTED, linewidth=1.0),
        capprops=dict(color=INK_MUTED, linewidth=1.0),
    )
    # color follows the entity: category index in its fixed order picks the slot
    for patch, cat in zip(bp["boxes"], cats):
        color = CATEGORICAL[order.index(cat) % len(CATEGORICAL)]
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)

    # n per category under each tick (uneven design: DIS=40, PHA=50 files)
    labels = [f"{c}\n(n={len(d)})" for c, d in zip(cats, data)]
    ax.set_xticks(range(1, len(cats) + 1))
    ax.set_xticklabels(labels, fontsize=8)

    if log_scale:
        vals = np.concatenate([d for d in data if len(d)])
        if (vals > 0).all():
            ax.set_yscale("log")
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_feature(df, col, label, log_scale):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, (group_col, group_label, order) in zip(axes, GROUP_DIMS):
        styled_boxplot(ax, df, group_col, order, col, log_scale)
        ax.set_xlabel(group_label, fontsize=9, color=INK_SECONDARY)
    axes[0].set_ylabel(label, fontsize=9)
    fig.suptitle(f"{col} - distribution by drone / interference / flight mode",
                 fontsize=11, color=INK_PRIMARY, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = RESULTS_DIR / f"box_{col}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_overview(df):
    """Compact grid: every feature boxplotted by drone model only."""
    ncols = 3
    nrows = int(np.ceil(len(FEATURES) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.4 * nrows))
    for ax, (col, label, log_scale) in zip(axes.flat, FEATURES):
        styled_boxplot(ax, df, "drone_id", DRONE_ORDER, col, log_scale)
        ax.set_title(label, fontsize=9, color=INK_SECONDARY, loc="left")
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes.flat[len(FEATURES):]:
        ax.set_visible(False)
    fig.suptitle("Summary features by drone model - overview",
                 fontsize=12, color=INK_PRIMARY, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = RESULTS_DIR / "overview_by_drone.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_summary()
    print(f"Loaded {len(df)} rows from {SUMMARY_PARQUET.resolve()}")

    for col, label, log_scale in FEATURES:
        out = plot_feature(df, col, label, log_scale)
        print(f"  wrote {out.resolve().relative_to(SCRIPT_DIR.parent.parent)}")
    out = plot_overview(df)
    print(f"  wrote {out.resolve().relative_to(SCRIPT_DIR.parent.parent)}")
    print("Done.")


if __name__ == "__main__":
    main()
