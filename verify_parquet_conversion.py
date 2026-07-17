### Verifies the .dat -> .parquet conversion is complete and lossless ###
### Ground truth is read from the same source as the conversion (zip archive by
### default, or an extracted .dat folder), reusing LoadDataTransferParquet so the
### source/path interpretation stays identical to the converter. ###

import random
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from load_data_transfer_parquet import LoadDataTransferParquet

# Ground-truth source: set SOURCE_ZIP to the archive to verify without extracting,
# or set it to None to verify against the extracted .dat folder (SRC_FOLDER) instead.
SOURCE_ZIP = Path("D:/DroneEDA/DroneDetect_V2.zip")
SRC_FOLDER = Path("D:/DroneEDA/DroneDetect_V2/DroneDetect_V2")
OUT = Path("D:/DroneEDA/DroneDetect_V2_parquet")
SAMPLE_SIZE = 20
SEED = 42


def build_source_sizes(loader: LoadDataTransferParquet) -> dict[str, int]:
    # map each relative .dat path -> its raw byte size, without reading file contents
    if loader.source_zip is not None:
        prefix = loader._get_zip_prefix()
        with zipfile.ZipFile(loader.source_zip) as zf:
            sizes = {}
            for info in zf.infolist():
                name = info.filename
                if name.lower().endswith(".dat") and not name.endswith("/"):
                    rel = name[len(prefix):] if name.startswith(prefix) else name
                    sizes[rel] = info.file_size
            return sizes
    return {rel: (loader.file_folder / rel).stat().st_size for rel in loader.find_dat_files()}


def fast_pass(loader, relative_paths, source_sizes):
    # complex64 = 8 bytes/sample (4 bytes I + 4 bytes Q)
    missing, mismatches, ok = [], [], 0
    for rel in relative_paths:
        out = loader.resolve_output_path(rel)
        if not out.exists():
            missing.append(rel)
            continue
        expected = source_sizes[rel] // 8
        actual = pq.ParquetFile(out).metadata.num_rows
        if actual != expected:
            mismatches.append((rel, expected, actual))
        else:
            ok += 1
    return ok, missing, mismatches


def bit_exact_check(loader, relative_path) -> bool:
    raw = np.frombuffer(loader._read_bytes(relative_path), dtype=np.float32)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    I_truth, Q_truth = raw[0::2], raw[1::2]

    out = loader.resolve_output_path(relative_path)
    df = pq.read_table(out, columns=["index", "I", "Q"]).to_pandas()

    index_ok = (
        df["index"].iloc[0] == 0
        and df["index"].iloc[-1] == len(df) - 1
        and df["index"].is_monotonic_increasing
    )
    I_ok = np.array_equal(df["I"].to_numpy(), I_truth)
    Q_ok = np.array_equal(df["Q"].to_numpy(), Q_truth)
    return index_ok and I_ok and Q_ok


def main():
    loader = LoadDataTransferParquet(
        file_folder=SRC_FOLDER,
        output_folder=OUT,
        source_zip=SOURCE_ZIP,
    )
    source_desc = f"zip:{loader.source_zip}" if loader.source_zip is not None else str(loader.file_folder)
    relative_paths = loader.find_dat_files()
    source_sizes = build_source_sizes(loader)
    print(f"Ground-truth source: {source_desc}")
    print(f"Total .dat files: {len(relative_paths)}")

    print("\n=== Pass 1: row-count check (metadata only, all files) ===")
    ok, missing, mismatches = fast_pass(loader, relative_paths, source_sizes)
    print(f"OK: {ok} | Missing parquet: {len(missing)} | Row mismatches: {len(mismatches)}")
    for rel in missing:
        print(f"  MISSING PARQUET: {rel}")
    for rel, expected, actual in mismatches:
        print(f"  ROW MISMATCH: {rel} expected={expected} actual={actual}")

    print(f"\n=== Pass 2: bit-exact spot check ({SAMPLE_SIZE} random files) ===")
    rng = random.Random(SEED)
    sample = rng.sample(relative_paths, min(SAMPLE_SIZE, len(relative_paths)))
    fail_count = 0
    for i, rel in enumerate(sample, start=1):
        passed = bit_exact_check(loader, rel)
        status = "OK" if passed else "FAIL"
        if not passed:
            fail_count += 1
        print(f"  [{i}/{len(sample)}] {status}: {rel}")

    print("\n=== Summary ===")
    all_ok = not missing and not mismatches and fail_count == 0
    print(f"Row-count check: {'PASS' if not missing and not mismatches else 'FAIL'}")
    print(f"Bit-exact spot check: {'PASS' if fail_count == 0 else f'FAIL ({fail_count} files)'}")
    print("OVERALL:", "LOSSLESS / COMPLETE" if all_ok else "ISSUES FOUND — see above")


if __name__ == "__main__":
    main()
