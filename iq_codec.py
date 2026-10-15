### Lossless block codec for the DroneDetect IQ parquet files ###
### The recorder is a 12-bit ADC: every float32 I/Q value is exactly k / 2048 with an integer k. ###
### Each channel is cut into blocks of 4096 samples. A block is stored either as k - base (base = ###
### block mean) or as first differences, whichever has the smaller spread, zigzag-coded and ###
### split into a low-byte and a high-byte plane. The high plane of a narrow block is all zero ###
### and stored as NULL; zstd does the entropy coding. Decoding is bit-exact. ###

from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCALE = 2048                     # float32 value = k / SCALE (exact: SCALE is a power of two)
BLOCK = 4096                     # samples per block (one parquet row)
CHUNK_BLOCKS = 256               # blocks per row group -> 1,048,576 samples
ZSTD_LEVEL = 19                  # encode-time cost only; decode speed is level independent
CHANNELS = ("I", "Q")
MAX_ABS_K = 1 << 14              # keeps ref / first within int16 and zigzag residuals within 16 bits
DELTA_MARGIN_BITS = 0.5          # delta mode must beat base mode by this many bits of spread

META_CODEC = b"iq_codec"
META_SAMPLES = b"iq_samples"
CODEC_ID = b"block-base-v1"
LEGACY_COLUMNS = ["I", "Q"]


class CodecError(ValueError):
    """Raised when data cannot be represented losslessly by the codec."""


def _column_names() -> list[str]:
    return [f"{c}_{f}" for c in CHANNELS for f in ("ref", "first", "mode", "lo", "hi")]


def build_schema(n_samples: int) -> pa.Schema:
    fields = []
    for c in CHANNELS:
        fields += [
            pa.field(f"{c}_ref", pa.int16()),
            pa.field(f"{c}_first", pa.int16()),
            pa.field(f"{c}_mode", pa.int8()),
            pa.field(f"{c}_lo", pa.binary(BLOCK)),
            pa.field(f"{c}_hi", pa.binary(BLOCK)),
        ]
    meta = {META_CODEC: CODEC_ID, META_SAMPLES: str(n_samples).encode()}
    return pa.schema(fields, metadata=meta)


def _to_k(x: np.ndarray) -> np.ndarray:
    """float32 samples -> int32 k, raising unless the round trip is bit-exact."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.size and not np.isfinite(x).all():
        raise CodecError("non-finite sample")
    if x.size and float(np.abs(x).max()) * SCALE > MAX_ABS_K:
        raise CodecError("sample magnitude out of range")
    k = np.rint(x * np.float32(SCALE)).astype(np.int32)
    back = k.astype(np.float32) / np.float32(SCALE)
    if not np.array_equal(back.view(np.uint32), x.view(np.uint32)):
        raise CodecError("sample is not on the k/2048 grid (or is -0.0)")
    return k


def _encode_channel(x: np.ndarray) -> dict:
    """Encode one channel of one chunk (len(x) <= CHUNK_BLOCKS*BLOCK) into per-block arrays."""
    k = _to_k(x)
    pad = (-len(k)) % BLOCK
    if pad:
        k = np.concatenate([k, np.full(pad, k[-1], np.int32)])   # edge padding, dropped on decode
    k = k.reshape(-1, BLOCK)
    # base mode: residual against the rounded block mean
    base = np.rint(k.mean(axis=1, dtype=np.float64)).astype(np.int32)
    r0 = k - base[:, None]
    # delta mode: first differences; the first sample of each block restarts from `first`
    d = np.zeros_like(k)
    d[:, 1:] = k[:, 1:] - k[:, :-1]
    delta = np.log2(d.std(axis=1) + 1e-9) + DELTA_MARGIN_BITS < np.log2(r0.std(axis=1) + 1e-9)
    r = np.where(delta[:, None], d, r0)
    u = (r << 1) ^ (r >> 31)                                      # zigzag: small |r| -> small u
    if int(u.max()) > 0xFFFF:
        raise CodecError("block spread exceeds 16 bits")
    return {
        "ref": np.where(delta, 0, base).astype(np.int16),
        "first": k[:, 0].astype(np.int16),
        "mode": delta.astype(np.int8),
        "lo": (u & 0xFF).astype(np.uint8),
        "hi": (u >> 8).astype(np.uint8),
    }


def _fixed_binary(rows: np.ndarray, null_rows: np.ndarray | None = None) -> pa.Array:
    n = rows.shape[0]
    validity, null_count = None, 0
    if null_rows is not None and null_rows.any():
        validity = pa.py_buffer(np.packbits(~null_rows, bitorder="little").tobytes())
        null_count = int(null_rows.sum())
    return pa.FixedSizeBinaryArray.from_buffers(
        pa.binary(BLOCK), n, [validity, pa.py_buffer(np.ascontiguousarray(rows).tobytes())], null_count)


def encode_chunk(i_chunk: np.ndarray, q_chunk: np.ndarray, schema: pa.Schema) -> pa.Table:
    if len(i_chunk) != len(q_chunk):
        raise CodecError("I and Q length mismatch")
    cols = []
    for c, x in zip(CHANNELS, (i_chunk, q_chunk)):
        e = _encode_channel(x)
        hi_zero = ~e["hi"].any(axis=1)
        cols += [pa.array(e["ref"]), pa.array(e["first"]), pa.array(e["mode"]),
                 _fixed_binary(e["lo"]), _fixed_binary(e["hi"], hi_zero)]
    return pa.Table.from_arrays(cols, schema=schema)


def _plane(col: pa.ChunkedArray, nb: int) -> tuple[np.ndarray, np.ndarray]:
    arr = col.combine_chunks()
    data = np.frombuffer(arr.buffers()[1], np.uint8, count=nb * BLOCK, offset=arr.offset * BLOCK)
    valid = arr.is_valid().to_numpy(zero_copy_only=False)
    return data.reshape(nb, BLOCK), valid


def decode_chunk(table: pa.Table) -> tuple[np.ndarray, np.ndarray]:
    """Decode one row group into float32 (I, Q); the last block may still hold edge padding."""
    nb = table.num_rows
    out = []
    for c in CHANNELS:
        lo, _ = _plane(table[f"{c}_lo"], nb)
        hi, hi_valid = _plane(table[f"{c}_hi"], nb)
        u = lo.astype(np.int32)
        u[hi_valid] |= hi[hi_valid].astype(np.int32) << 8      # null slots hold garbage: never read them
        ref = table[f"{c}_ref"].to_numpy().astype(np.int32)
        first = table[f"{c}_first"].to_numpy().astype(np.int32)
        delta = table[f"{c}_mode"].to_numpy().astype(bool)
        r = (u >> 1) ^ -(u & 1)                                  # undo zigzag
        u = r + ref[:, None]
        if delta.any():
            d = r[delta]
            d[:, 0] = 0
            u[delta] = first[delta][:, None] + np.cumsum(d, axis=1, dtype=np.int32)
        out.append((u.astype(np.float32) / np.float32(SCALE)).reshape(-1))
    return out[0], out[1]


def is_coded(schema: pa.Schema) -> bool:
    return bool(schema.metadata) and schema.metadata.get(META_CODEC) == CODEC_ID


class IQWriter:
    """Stream (I, Q) chunks into one codec parquet file. All chunks but the last must be CHUNK_BLOCKS*BLOCK long."""

    def __init__(self, path: Path, n_samples: int):
        self.n_samples, self.written, self._closed_short = n_samples, 0, False
        self.schema = build_schema(n_samples)
        self._w = pq.ParquetWriter(path, self.schema, compression="zstd", compression_level=ZSTD_LEVEL,
                                   use_dictionary=False, write_statistics=False)

    def write_chunk(self, i: np.ndarray, q: np.ndarray) -> None:
        step = CHUNK_BLOCKS * BLOCK
        if self._closed_short:
            raise CodecError("only the last chunk may be shorter than a full chunk")
        if len(i) > step:
            raise CodecError("chunk larger than a row group")
        self._closed_short = len(i) < step
        self._w.write_table(encode_chunk(i, q, self.schema))
        self.written += len(i)

    def close(self) -> None:
        self._w.close()
        if self.written != self.n_samples:
            raise CodecError(f"wrote {self.written} samples, declared {self.n_samples}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self._w.close()
        else:
            self.close()


def write_iq_parquet(path: Path, i: np.ndarray, q: np.ndarray) -> None:
    """Write whole I/Q (float32) arrays to `path` in the codec layout."""
    if len(q) != len(i):
        raise CodecError("I and Q length mismatch")
    step = CHUNK_BLOCKS * BLOCK
    with IQWriter(path, len(i)) as w:
        for s in range(0, len(i), step):
            w.write_chunk(i[s:s + step], q[s:s + step])


def num_samples(path: Path) -> int:
    with pq.ParquetFile(path) as pf:
        if is_coded(pf.schema_arrow):
            return int(pf.schema_arrow.metadata[META_SAMPLES])
        return pf.metadata.num_rows


def iter_iq(path: Path) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield float32 (I, Q) arrays row group by row group, in source order."""
    with pq.ParquetFile(path) as pf:
        if not is_coded(pf.schema_arrow):
            if pf.schema_arrow.names != LEGACY_COLUMNS:
                raise CodecError(f"unexpected columns {pf.schema_arrow.names}")
            for g in range(pf.metadata.num_row_groups):
                t = pf.read_row_group(g, columns=LEGACY_COLUMNS)
                yield t["I"].to_numpy(), t["Q"].to_numpy()
            return
        remaining = int(pf.schema_arrow.metadata[META_SAMPLES])
        for g in range(pf.metadata.num_row_groups):
            i, q = decode_chunk(pf.read_row_group(g))
            take = min(len(i), remaining)
            remaining -= take
            yield i[:take], q[:take]


def read_iq(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the whole recording as float32 (I, Q) arrays."""
    n = num_samples(path)
    i, q = np.empty(n, np.float32), np.empty(n, np.float32)
    pos = 0
    for ci, cq in iter_iq(path):
        i[pos:pos + len(ci)] = ci
        q[pos:pos + len(cq)] = cq
        pos += len(ci)
    if pos != n:
        raise CodecError(f"decoded {pos} samples, expected {n}")
    return i, q
