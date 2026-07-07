import numpy as np
import torch
from torch.utils.data import Dataset

from neurocardio.config import Config
from neurocardio.data.preprocess import bandpass_filter, normalize
from neurocardio.data.records import load_record
from neurocardio.data.segment import beat_rr_features, segment_beats


class ECGBeatDataset(Dataset):
    """Holds pre-segmented beats [N, L] and integer AAMI labels [N].

    If `transform` is given it is applied to each beat (numpy [L]) and should
    return a tensor; otherwise the beat is returned as a float32 tensor [L].
    """

    def __init__(self, beats: np.ndarray, labels: np.ndarray, transform=None):
        assert len(beats) == len(labels)
        self.beats = np.asarray(beats, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        beat = self.beats[idx]
        y = int(self.labels[idx])
        if self.transform is not None:
            x = self.transform(beat)
        else:
            x = torch.from_numpy(beat)
        return x, y


def build_split(config: Config, record_ids):
    """Load each record, preprocess, segment into AAMI beats, and concatenate."""
    all_beats, all_labels = [], []
    for rid in record_ids:
        rec = load_record(config.data.data_dir, rid, config.data.lead_index)
        sig = normalize(
            bandpass_filter(
                rec.signal,
                fs=rec.fs,
                low=config.data.bandpass_low,
                high=config.data.bandpass_high,
                order=config.data.filter_order,
            )
        )
        beats, labels = segment_beats(
            sig,
            rec.ann_samples,
            rec.ann_symbols,
            window_before=config.data.window_before,
            window_after=config.data.window_after,
        )
        if len(beats):
            all_beats.append(beats)
            all_labels.append(labels)
    if not all_beats:
        return (
            np.zeros((0, config.data.window_before + config.data.window_after)),
            np.zeros((0,), dtype=np.int64),
        )
    return np.concatenate(all_beats), np.concatenate(all_labels)


def build_split_rr(config: Config, record_ids):
    """Like build_split, but also returns per-beat RR-interval features [N, 3]
    (patient-normalized timing cues) aligned with the beats and labels."""
    all_beats, all_labels, all_rr = [], [], []
    for rid in record_ids:
        rec = load_record(config.data.data_dir, rid, config.data.lead_index)
        sig = normalize(
            bandpass_filter(
                rec.signal,
                fs=rec.fs,
                low=config.data.bandpass_low,
                high=config.data.bandpass_high,
                order=config.data.filter_order,
            )
        )
        beats, labels = segment_beats(
            sig,
            rec.ann_samples,
            rec.ann_symbols,
            window_before=config.data.window_before,
            window_after=config.data.window_after,
        )
        rr = beat_rr_features(
            rec.ann_samples,
            rec.ann_symbols,
            len(sig),
            window_before=config.data.window_before,
            window_after=config.data.window_after,
        )
        if len(beats):
            all_beats.append(beats)
            all_labels.append(labels)
            all_rr.append(rr)
    if not all_beats:
        w = config.data.window_before + config.data.window_after
        return np.zeros((0, w)), np.zeros((0,), dtype=np.int64), np.zeros((0, 3))
    return np.concatenate(all_beats), np.concatenate(all_labels), np.concatenate(all_rr)
