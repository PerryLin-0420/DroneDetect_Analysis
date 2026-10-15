### Verifies the .dat -> .parquet conversion is complete and lossless ###
### Ground truth is read from the same source as the conversion (zip archive by
### default, or an extracted .dat folder), reusing LoadDataTransferParquet so the
### source/path interpretation stays identical to the converter. ###

import random
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import iq_codec
from load_data_transfer_parquet import LoadDataTransferParquet

# Ground-truth source: set SOURCE_ZIP to the archive to verify without extracting,
# or set it to None to verify against the extracted .dat folder (SRC_FOLDER) instead.
SOURCE_ZIP = Path("D:/DroneEDA/DroneDetect_V2.zip")
SRC_FOLDER = Path("D:/DroneEDA/DroneDetect_V2/DroneDetect_V2")
OUT = Path("D:/DroneEDA/DroneDetect_V2_parquet")
SAMPLE_SIZE = 20
SEED = 42
# a file is either in the block-codec layout (schema from iq_codec) or plain float32 I/Q
EXPECTED_CODEC_COLUMNS = iq_codec.build_schema(0).names
EXPECTED_PLAIN_COLUMNS = iq_codec.LEGACY_COLUMNS


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
    missing, mismatches, schema_bad, ok = [], [], [], 0
    for rel in relative_paths:
        out = loader.resolve_output_path(rel)
        if not out.exists():
            missing.append(rel)
            continue
        expected = source_sizes[rel] // 8
        with pq.ParquetFile(out) as pf:
            names = pf.schema_arrow.names
            coded = iq_codec.is_coded(pf.schema_arrow)
            actual = iq_codec.num_samples(out)
            if names != (EXPECTED_CODEC_COLUMNS if coded else EXPECTED_PLAIN_COLUMNS):
                schema_bad.append((rel, names))
            elif actual != expected:
                mismatches.append((rel, expected, actual))
            else:
                ok += 1
    return ok, missing, mismatches, schema_bad


def bit_exact_check(loader, relative_path) -> bool:
    raw = np.frombuffer(loader._read_bytes(relative_path), dtype=np.float32)
    if raw.size % 2 != 0:
        raw = raw[:-1]
    I_truth, Q_truth = raw[0::2], raw[1::2]

    out = loader.resolve_output_path(relative_path)

    # stream the decoded file chunk by chunk and compare against the source in order; a match on
    # every chunk also proves sample order (the row position is the sample index).
    # compare as uint32 bit patterns so NaN payloads / -0.0 are verified bit-exact too
    pos = 0
    for i_chunk, q_chunk in iq_codec.iter_iq(out):
        n = len(i_chunk)
        if pos + n > len(I_truth):
            return False
        if not (np.array_equal(i_chunk.view(np.uint32), I_truth[pos:pos + n].view(np.uint32))
                and np.array_equal(q_chunk.view(np.uint32), Q_truth[pos:pos + n].view(np.uint32))):
            return False
        pos += n
    return pos == len(I_truth)


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

    print("\n=== Pass 1: row-count + schema check (metadata only, all files) ===")
    ok, missing, mismatches, schema_bad = fast_pass(loader, relative_paths, source_sizes)
    print(f"OK: {ok} | Missing parquet: {len(missing)} | Row mismatches: {len(mismatches)} | Schema mismatches: {len(schema_bad)}")
    for rel in missing:
        print(f"  MISSING PARQUET: {rel}")
    for rel, expected, actual in mismatches:
        print(f"  ROW MISMATCH: {rel} expected={expected} actual={actual}")
    for rel, names in schema_bad:
        print(f"  SCHEMA MISMATCH: {rel} columns={names} expected={EXPECTED_CODEC_COLUMNS} or {EXPECTED_PLAIN_COLUMNS}")

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
    all_ok = not missing and not mismatches and not schema_bad and fail_count == 0
    print(f"Row-count check: {'PASS' if not missing and not mismatches else 'FAIL'}")
    print(f"Schema check: {'PASS' if not schema_bad else f'FAIL ({len(schema_bad)} files)'}")
    print(f"Bit-exact spot check: {'PASS' if fail_count == 0 else f'FAIL ({fail_count} files)'}")
    print("OVERALL:", "LOSSLESS / COMPLETE" if all_ok else "ISSUES FOUND — see above")


if __name__ == "__main__":
    main()
