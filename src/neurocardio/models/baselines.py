import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """1-D CNN baseline on raw beats [B, L]."""

    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.unsqueeze(1))  # [B, 1, L]
        return self.head(h.squeeze(-1))


class LSTMClassifier(nn.Module):
    """LSTM baseline on raw beats [B, L] (treated as length-L, 1-feature seq)."""

    def __init__(self, n_classes: int = 5, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x.unsqueeze(-1))  # [B, L, 1]
        return self.head(out[:, -1, :])
