import numpy as np
import torch
import wfdb

from neurocardio.config import Config
from neurocardio.data.dataset import ECGBeatDataset, build_split


def test_dataset_returns_beat_and_label():
    beats = np.random.randn(5, 256).astype(np.float64)
    labels = np.array([0, 1, 2, 3, 4])
    ds = ECGBeatDataset(beats, labels)
    x, y = ds[2]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (256,)
    assert int(y) == 2
    assert len(ds) == 5


def test_dataset_applies_transform():
    beats = np.zeros((2, 4), dtype=np.float64)
    labels = np.array([0, 1])

    def to_two_channels(beat):
        return torch.zeros((beat.shape[0], 2), dtype=torch.float32)

    ds = ECGBeatDataset(beats, labels, transform=to_two_channels)
    x, _ = ds[0]
    assert x.shape == (4, 2)


def _write_two_beat_record(dirpath, record_id):
    fs = 360
    n = fs * 5  # long enough that both beats (incl. the ±128 window at 1500) fit
    sig = np.zeros((n, 1))
    for center in (500, 1500):
        sig[center - 2 : center + 3, 0] = [0.2, 0.6, 1.0, 0.6, 0.2]
    wfdb.wrsamp(
        record_id, fs=fs, units=["mV"], sig_name=["MLII"], p_signal=sig, write_dir=str(dirpath)
    )
    wfdb.wrann(
        record_id, "atr", sample=np.array([500, 1500]), symbol=["N", "V"], write_dir=str(dirpath)
    )


def test_build_split_produces_beats_and_labels(tmp_path):
    _write_two_beat_record(tmp_path, "900")
    cfg = Config()
    cfg.data.data_dir = str(tmp_path)
    beats, labels = build_split(cfg, record_ids=["900"])
    assert beats.shape[1] == 256
    assert beats.shape[0] == 2
    assert set(labels.tolist()) == {0, 2}  # N and VEB
