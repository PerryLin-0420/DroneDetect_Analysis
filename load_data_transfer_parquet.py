### Produced by Carolyn J. Swinney and John C. Woods for use with the 'DroneDetect' dataset ###

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from attrs import define, field

SAMPLE_RATE_HZ = 60_000_000  # complex samples/sec (1.2e8 samples / 2s, per dataset author)


@define
class LoadDataTransferParquet:
    file_folder: Path = field(default=Path("D:/DroneEDA/DroneDetect_V2/DroneDetect_V2"))
    output_folder: Path = field(default=Path("D:/DroneEDA/DroneDetect_V2_parquet"))
    # if set, .dat files are read directly from this zip archive instead of file_folder,
    # so a fresh download never needs to be extracted to disk just to be converted.
    # this is the primary source: the extracted .dat folder can be deleted once parquet exists,
    # and re-downloading the zip here is enough to regenerate everything.
    source_zip: Path | None = field(default=Path("D:/DroneEDA/DroneDetect_V2.zip"))
    # cached top-level folder to strip from zip entries (e.g. "DroneDetect_V2/"), so output
    # paths mirror CLEAN/BLUE/WIFI/BOTH directly regardless of how the zip wraps the dataset
    _zip_prefix: str | None = field(default=None, init=False, repr=False)

    def _get_zip_prefix(self) -> str:
        if self._zip_prefix is None:
            with zipfile.ZipFile(self.source_zip) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".dat") and not n.endswith("/")]
            top_level = {n.split("/", 1)[0] for n in names if "/" in n}
            self._zip_prefix = next(iter(top_level)) + "/" if len(top_level) == 1 else ""
        return self._zip_prefix

    def find_dat_files(self) -> list[str]:
        # every .dat as a POSIX-style path relative to the dataset root, sorted for stable ordering
        if self.source_zip is not None:
            with zipfile.ZipFile(self.source_zip) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".dat") and not n.endswith("/")]
            prefix = self._get_zip_prefix()
            return sorted(n[len(prefix):] if n.startswith(prefix) else n for n in names)
        return sorted(p.relative_to(self.file_folder).as_posix() for p in self.file_folder.rglob("*.dat"))

    def resolve_output_path(self, relative_path: str) -> Path:
        # mirror the source structure under output_folder, swapping .dat for .parquet
        return (self.output_folder / relative_path).with_suffix(".parquet")

    def _read_bytes(self, relative_path: str) -> bytes:
        if self.source_zip is not None:
            with zipfile.ZipFile(self.source_zip) as zf:
                return zf.read(self._get_zip_prefix() + relative_path)
        return (self.file_folder / relative_path).read_bytes()

    def load_data(self, relative_path: str) -> np.ndarray:
        try:
            # read the ENTIRE file (no count cap) so no trailing samples are dropped;
            # many files exceed 240M floats and would otherwise be truncated
            data = np.frombuffer(self._read_bytes(relative_path), dtype=np.float32)
            # interleaved I/Q floats -> complex64; guard against odd-length (misaligned) files
            if data.size % 2 != 0:
                print(f"Warning: {relative_path} has an odd float count ({data.size}); dropping last float")
                data = data[:-1]
            # bit-exact reinterpretation of the raw bytes, no value change, no normalisation
            data = data.view(np.complex64)
            return data
        except Exception as e:
            print(f"Error loading data from {relative_path}: {e}")
            return None

    def transfer_data(self, data: np.ndarray) -> pd.DataFrame:
        try:
            df = pd.DataFrame({
                'index': np.arange(len(data), dtype=np.int64), # preserves sample order for later time-based analysis
                'I': data.real,
                'Q': data.imag,
            })
            return df
        except Exception as e:
            print(f"Error transferring data: {e}")
            return None

    def save_to_parquet(self, df: pd.DataFrame, output_path: Path):
        try:
            # zstd is lossless (like all parquet codecs); chosen over snappy for a better
            # size/IO trade-off. Float values are stored bit-exact regardless of codec.
            df.to_parquet(output_path, engine='pyarrow', index=False, compression='zstd')
        except Exception as e:
            print(f"Error saving to parquet: {e}")

    def convert_all(self):
        relative_paths = self.find_dat_files()
        total = len(relative_paths)
        source_desc = f"zip:{self.source_zip}" if self.source_zip is not None else str(self.file_folder)
        print(f"Found {total} .dat files under {source_desc}")

        for idx, relative_path in enumerate(relative_paths, start=1):
            output_path = self.resolve_output_path(relative_path)

            if output_path.exists():
                print(f"[{idx}/{total}] skip (already exists): {relative_path}")
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{idx}/{total}] converting: {relative_path}")

            data = self.load_data(relative_path)
            if data is None:
                continue

            df = self.transfer_data(data)
            del data  # free the ~1GB complex array before writing
            if df is None:
                continue

            self.save_to_parquet(df, output_path)
            del df  # free the DataFrame before moving to the next file

        print("Done.")


if __name__ == "__main__":
    # default: convert directly from the zip archive (source_zip) to the parquet folder,
    # skipping any parquet that already exists
    loader = LoadDataTransferParquet()
    loader.convert_all()
    # To convert from an already-extracted .dat folder instead of the zip:
    # loader = LoadDataTransferParquet(source_zip=None)
    # loader.convert_all()
