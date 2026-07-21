### CNN stage input: 50 ms segments -> pooled log-magnitude spectrograms ###
### Same segmentation as the PSD stage (40 x 50 ms per recording). Per       ###
### segment: STFT (nperseg 1024, hop 512, two-sided, fftshifted), power      ###
### pooled to a compact 256(F) x 128(T) grid, then dB. float16 output        ###
### (~1 GB total) so the whole tensor trains in RAM on CPU. Per-segment      ###
### standardization (gain removal) happens at train time, not here.          ###

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.signal import spectrogram

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
PARQUET_ROOT = SCRIPT_DIR / ".." / ".." / "DroneDetect_V2_parquet"
RESULTS_DIR = SCRIPT_DIR / ".." / "results"

SAMPLE_RATE_HZ = 60_000_000
SEGMENT_SAMPLES = 3_000_000      # 50 ms
NPERSEG = 1024
HOP = 512
# frequency bins: optional CLI arg (256 default); 512/1024 for a higher-resolution run
FREQ_BINS = int(sys.argv[1]) if len(sys.argv) > 1 else 256   # 1024 -> N (mean-pool x 1024/N)
TIME_BINS = 128                  # ~5858 frames -> 128 (mean-pool x45, ~390 us/bin)
CLIP_THRESHOLD = 0.99
_SUFFIX = "" if FREQ_BINS == 256 else f"_{FREQ_BINS}"  # keep 256 filenames unchanged

OUT_SPECS = RESULTS_DIR / f"spectrograms{_SUFFIX}.npy"        # (N, FREQ_BINS, 128) float16
OUT_META = RESULTS_DIR / f"spectrogram_meta{_SUFFIX}.parquet" # N rows, same order

INTERFERENCE_TEXT = {"CLEAN": "clean", "BLUE": "bluetooth", "WIFI": "wifi", "BOTH": "bluetooth_wifi"}


def parse_metadata(relative_path: str) -> dict:
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


def segment_spectrogram(iq: np.ndarray) -> np.ndarray:
    """Return pooled log-power spectrogram, shape (FREQ_BINS, TIME_BINS), float16."""
    _, _, sxx = spectrogram(iq, fs=SAMPLE_RATE_HZ, nperseg=NPERSEG,
                            noverlap=NPERSEG - HOP, return_onesided=False,
                            detrend=False, mode="psd")
    sxx = np.fft.fftshift(sxx, axes=0)          # (1024, ~5858) linear power

    # mean-pool in the linear power domain, then dB
    f_factor = sxx.shape[0] // FREQ_BINS
    t_frames = (sxx.shape[1] // TIME_BINS) * TIME_BINS
    t_factor = t_frames // TIME_BINS
    pooled = sxx[:, :t_frames].reshape(FREQ_BINS, f_factor, TIME_BINS, t_factor).mean(axis=(1, 3))
    return (10.0 * np.log10(pooled + 1e-20)).astype(np.float16)


def process_file(pq_path: Path, relative_path: str):
    meta = parse_metadata(relative_path)
    table = pq.read_table(pq_path, columns=["I", "Q"])
    i = table.column("I").to_numpy(zero_copy_only=False)
    q = table.column("Q").to_numpy(zero_copy_only=False)
    iq = i.astype(np.float32) + 1j * q.astype(np.float32)
    del table, i, q

    n_segments = len(iq) // SEGMENT_SAMPLES
    specs, rows = [], []
    for s in range(n_segments):
        seg = iq[s * SEGMENT_SAMPLES:(s + 1) * SEGMENT_SAMPLES]
        clip = float(np.mean((np.abs(seg.real) >= CLIP_THRESHOLD) | (np.abs(seg.imag) >= CLIP_THRESHOLD)))
        specs.append(segment_spectrogram(seg))
        rows.append({**meta, "segment_index": s, "seg_clip_ratio": clip})
    return specs, rows


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PARQUET_ROOT.rglob("*.parquet"))
    total = len(files)
    print(f"Found {total} parquet files; output grid = {FREQ_BINS}(F) x {TIME_BINS}(T)")

    all_specs, all_rows = [], []
    t0 = time.time()
    for idx, pq_path in enumerate(files, start=1):
        relative_path = pq_path.relative_to(PARQUET_ROOT).as_posix()
        specs, rows = process_file(pq_path, relative_path)
        all_specs.extend(specs)
        all_rows.extend(rows)
        elapsed = time.time() - t0
        eta = elapsed / idx * (total - idx)
        print(f"[{idx}/{total}] {relative_path}  (elapsed {elapsed:.0f}s, eta {eta:.0f}s)")

    specs = np.stack(all_specs)
    np.save(OUT_SPECS, specs)
    pd.DataFrame(all_rows).to_parquet(OUT_META)
    print(f"\nWrote {specs.shape} float16 -> {OUT_SPECS.resolve()} "
          f"({specs.nbytes / 1e9:.2f} GB)")
    print(f"Wrote {len(all_rows)} meta rows -> {OUT_META.resolve()}")


if __name__ == "__main__":
    main()
