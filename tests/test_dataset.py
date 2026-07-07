import numpy as np
import torch
import wfdb

from neurocardio.config import Config
from neurocardio.data.dataset import (
    ECGBeatDataset,
    build_external_split,
    build_external_split_rr,
    build_split,
)


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


def _write_external_record(dirpath, record_id, fs):
    """A record at an arbitrary rate (e.g. svdb 128 Hz) with three annotated beats."""
    n = fs * 6
    sig = np.zeros((n, 1))
    samples = [int(fs * 1.5), int(fs * 3.0), int(fs * 4.5)]
    for c in samples:
        sig[c - 2 : c + 3, 0] = [0.2, 0.6, 1.0, 0.6, 0.2]
    wfdb.wrsamp(
        record_id, fs=fs, units=["mV"], sig_name=["II"], p_signal=sig, write_dir=str(dirpath)
    )
    wfdb.wrann(
        record_id, "atr", sample=np.array(samples), symbol=["N", "V", "A"], write_dir=str(dirpath)
    )


def test_build_external_split_rr_resamples_segments_and_aligns(tmp_path):
    # Native 128 Hz record; config targets 360 Hz. Windows must come out at 360-Hz size.
    _write_external_record(tmp_path, "800", fs=128)
    (tmp_path / "RECORDS").write_text("800\n")
    cfg = Config()  # fs=360, window 128+128
    beats, labels, rr = build_external_split_rr(cfg, tmp_path)
    assert beats.shape == (3, 256)  # resampled onto the 360-Hz grid
    assert set(labels.tolist()) == {0, 1, 2}  # N, SVEB, VEB
    assert rr.shape == (3, 3)  # one RR-feature row per kept beat


def test_build_external_split_matches_rr_variant_beats(tmp_path):
    _write_external_record(tmp_path, "800", fs=128)
    cfg = Config()
    beats, labels = build_external_split(cfg, tmp_path, record_ids=["800"])
    assert beats.shape == (3, 256)
    assert len(labels) == 3
