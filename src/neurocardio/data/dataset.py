import numpy as np
import torch
from torch.utils.data import Dataset


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
