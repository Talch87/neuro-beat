import numpy as np
import torch
from torch.utils.data import DataLoader

from neurocardio.data.dataset import ECGBeatDataset
from neurocardio.eval.evaluate import evaluate


class _ConstModel(torch.nn.Module):
    """Always predicts class 0 (for a deterministic test)."""

    def forward(self, x):
        b = x.shape[0]
        out = torch.zeros(b, 3)
        out[:, 0] = 1.0
        return out


def test_evaluate_returns_cm_and_metrics():
    beats = np.zeros((6, 8), dtype=np.float32)
    labels = np.array([0, 0, 1, 1, 2, 2])
    ds = ECGBeatDataset(beats, labels)
    loader = DataLoader(ds, batch_size=3)
    result = evaluate(_ConstModel(), loader, classes=["N", "SVEB", "VEB"])
    assert result["confusion"].shape == (3, 3)
    assert result["metrics"]["per_class"]["N"]["sensitivity"] == 1.0
    assert result["metrics"]["per_class"]["VEB"]["sensitivity"] == 0.0
