### Stage-1 feature extraction: 50 ms segments -> normalized Welch PSD ###
### Reads the 390 IQ parquet files, slices each recording into 50 ms segments ###
### (3M complex samples @ 60 MS/s), computes a per-segment two-sided Welch    ###
### PSD, normalizes it (total power = 1, then dB) so the gain confound is     ###
### removed at the representation level, and stores one row per segment.      ###
### Output is small (~40 segments/file, 1024 bins) and feeds the baseline     ###
### classifiers (LDA / XGBoost); raw IQ is never needed again downstream.     ###

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import welch

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
PARQUET_ROOT = SCRIPT_DIR / ".." / ".." / "DroneDetect_V2_parquet"
RESULTS_DIR = SCRIPT_DIR / ".." / "results"
OUT_PARQUET = RESULTS_DIR / "psd_features.parquet"

SAMPLE_RATE_HZ = 60_000_000
SEGMENT_SAMPLES = 3_000_000          # 50 ms @ 60 MS/s -> 40 segments per 2 s recording
NPERSEG = 1024                       # PSD bins; freq resolution ~58.6 kHz
CLIP_THRESHOLD = 0.99                # same full-scale definition as the summary table

INTERFERENCE_TEXT = {"CLEAN": "clean", "BLUE": "bluetooth", "WIFI": "wifi", "BOTH": "bluetooth_wifi"}


def parse_metadata(relative_path: str) -> dict:
    # relative_path like "CLEAN/MIN_ON/MIN_0000_00.parquet"; drone_id from FOLDER
    parts = relative_path.split("/")
    drone_id, flight_mode = parts[1].split("_", 1)
    stem = Path(parts[-1]).stem
    code_iiff, run = stem.split("_")[1], stem.split("_")[2]
    return {
        "relative_path": relative_path,
        "drone_id": drone_id,
        "interference": INTERFERENCE_TEXT[parts[0]],
        "flight_mode": flight_mode,
        "run_index": int(run),
    }


def segment_psd(iq: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return (normalized PSD in dB, segment RMS, segment clip ratio)."""
    rms = float(np.sqrt(np.mean(iq.real**2 + iq.imag**2)))
    clip = float(np.mean((np.abs(iq.real) >= CLIP_THRESHOLD) | (np.abs(iq.imag) >= CLIP_THRESHOLD)))

    # two-sided PSD for complex baseband, shifted to monotonic frequency axis
    _, psd = welch(iq, fs=SAMPLE_RATE_HZ, nperseg=NPERSEG, noverlap=NPERSEG // 2,
                   return_onesided=False, detrend=False)
    psd = np.fft.fftshift(psd)
    # normalize total power to 1 -> gain-invariant spectral *shape*, then dB
    psd = psd / psd.sum()
    psd_db = 10.0 * np.log10(psd + 1e-20)
    return psd_db.astype(np.float32), rms, clip


def process_file(pq_path: Path, relative_path: str) -> list[dict]:
    meta = parse_metadata(relative_path)
    table = pq.read_table(pq_path, columns=["I", "Q"])
    i = table.column("I").to_numpy(zero_copy_only=False)
    q = table.column("Q").to_numpy(zero_copy_only=False)
    iq = i.astype(np.float32) + 1j * q.astype(np.float32)
    del table, i, q

    n_segments = len(iq) // SEGMENT_SAMPLES
    rows = []
    for s in range(n_segments):
        seg = iq[s * SEGMENT_SAMPLES:(s + 1) * SEGMENT_SAMPLES]
        psd_db, rms, clip = segment_psd(seg)
        rows.append({**meta, "segment_index": s, "seg_rms": rms,
                     "seg_clip_ratio": clip, "psd_db": psd_db})
    return rows


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PARQUET_ROOT.rglob("*.parquet"))
    total = len(files)
    print(f"Found {total} parquet files; segment = {SEGMENT_SAMPLES} samples "
          f"({SEGMENT_SAMPLES / SAMPLE_RATE_HZ * 1000:.0f} ms), PSD bins = {NPERSEG}")

    all_rows = []
    t0 = time.time()
    for idx, pq_path in enumerate(files, start=1):
        relative_path = pq_path.relative_to(PARQUET_ROOT).as_posix()
        all_rows.extend(process_file(pq_path, relative_path))
        elapsed = time.time() - t0
        eta = elapsed / idx * (total - idx)
        print(f"[{idx}/{total}] {relative_path}  (elapsed {elapsed:.0f}s, eta {eta:.0f}s)")

    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_PARQUET, compression="zstd")
    print(f"\nWrote {len(df)} segment rows to {OUT_PARQUET.resolve()}")


if __name__ == "__main__":
    main()
