from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb


@dataclass
class Record:
    record_id: str
    signal: np.ndarray  # 1-D single lead, shape [n_samples]
    fs: int
    ann_samples: np.ndarray  # int sample indices of beat annotations
    ann_symbols: list[str]  # wfdb beat symbols aligned to ann_samples


def load_record(record_dir, record_id: str, lead_index: int = 0) -> Record:
    base = str(Path(record_dir) / record_id)
    rec = wfdb.rdrecord(base)
    ann = wfdb.rdann(base, "atr")
    signal = np.asarray(rec.p_signal[:, lead_index], dtype=np.float64)
    return Record(
        record_id=record_id,
        signal=signal,
        fs=int(rec.fs),
        ann_samples=np.asarray(ann.sample, dtype=int),
        ann_symbols=list(ann.symbol),
    )
