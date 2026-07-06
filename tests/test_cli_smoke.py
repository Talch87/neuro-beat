import numpy as np
import torch
from torch.utils.data import DataLoader

from neurocardio.data.dataset import ECGBeatDataset
from neurocardio.data.segment import AAMI_CLASSES
from neurocardio.deploy.energy import spike_stats
from neurocardio.encoding.delta import delta_encode
from neurocardio.eval.evaluate import evaluate
from neurocardio.models.snn import SNNClassifier
from neurocardio.train.loop import set_seed, train


def test_end_to_end_synthetic_pipeline():
    set_seed(0)
    # synthetic separable beats: class 0 flat, class 2 has a step (VEB stand-in)
    n = 40
    beats = np.zeros((n, 256), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if i % 2 == 0:
            beats[i, 120:136] = 1.0
            labels[i] = 0
        else:
            beats[i, 120:200] = 1.0
            labels[i] = 2

    def transform(beat):
        return torch.from_numpy(delta_encode(beat, threshold=0.5))

    ds = ECGBeatDataset(beats, labels, transform=transform)
    loader = DataLoader(ds, batch_size=10, shuffle=True)
    model = SNNClassifier(in_features=2, hidden=32, n_classes=5)
    history = train(model, loader, loader, epochs=30, lr=0.02)
    # the SNN + delta + train integration must actually LEARN (loss decreases) --
    # this is the guard that the membrane-potential readout works end-to-end
    assert history["train_loss"][-1] < history["train_loss"][0]
    result = evaluate(model, loader, classes=AAMI_CLASSES)
    assert 0.0 <= result["metrics"]["overall_accuracy"] <= 1.0
    stats = spike_stats(model, next(iter(loader))[0][:1])
    assert stats["synops"] >= 0
