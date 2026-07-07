from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

from neurocardio.data.preprocess import resample_signal


@dataclass
class Record:
    record_id: str
    signal: np.ndarray  # 1-D single lead, shape [n_samples]
    fs: int
    ann_samples: np.ndarray  # int sample indices of beat annotations
    ann_symbols: list[str]  # wfdb beat symbols aligned to ann_samples


def load_record(
    record_dir, record_id: str, lead_index: int = 0, target_fs: int | None = None
) -> Record:
    """Load one WFDB record and its beat annotations as a single-lead Record.

    If target_fs is given and differs from the record's native rate, the signal is
    resampled and the annotation sample positions are rescaled to the target grid, so
    downstream sample-indexed windows (segment_beats) apply uniformly across databases
    recorded at different rates."""
    base = str(Path(record_dir) / record_id)
    rec = wfdb.rdrecord(base)
    ann = wfdb.rdann(base, "atr")
    native_fs = int(rec.fs)
    signal = np.asarray(rec.p_signal[:, lead_index], dtype=np.float64)
    ann_samples = np.asarray(ann.sample, dtype=int)
    fs = native_fs
    if target_fs is not None and target_fs != native_fs:
        signal = resample_signal(signal, native_fs, target_fs)
        ann_samples = np.round(ann_samples * target_fs / native_fs).astype(int)
        fs = target_fs
    return Record(
        record_id=record_id,
        signal=signal,
        fs=fs,
        ann_samples=ann_samples,
        ann_symbols=list(ann.symbol),
    )


def list_records(record_dir) -> list[str]:
    """Record ids in a downloaded WFDB database directory.

    Prefers the PhysioNet RECORDS index file; falls back to globbing .hea headers.
    Lets cross-database evaluation enumerate svdb/INCART without hardcoding ids."""
    record_dir = Path(record_dir)
    records_file = record_dir / "RECORDS"
    if records_file.exists():
        ids = [line.strip() for line in records_file.read_text().splitlines() if line.strip()]
        if ids:
            return [Path(r).name for r in ids]
    return sorted(p.stem for p in record_dir.glob("*.hea"))
