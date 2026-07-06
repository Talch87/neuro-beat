import numpy as np
import torch

from neurocardio.data.dataset import ECGBeatDataset


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
