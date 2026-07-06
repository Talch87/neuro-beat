import torch

from neurocardio.models.baselines import CNN1D, LSTMClassifier


def test_cnn_output_shape():
    model = CNN1D(n_classes=5)
    x = torch.randn(4, 256)  # [B, L] raw beat
    out = model(x)
    assert out.shape == (4, 5)


def test_lstm_output_shape():
    model = LSTMClassifier(n_classes=5, hidden=16)
    x = torch.randn(4, 256)  # [B, L] raw beat
    out = model(x)
    assert out.shape == (4, 5)
