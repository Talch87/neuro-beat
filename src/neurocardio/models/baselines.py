import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """1-D CNN baseline on raw beats [B, L], optionally with n_rr RR features
    concatenated before the classifier head (for a fair match to the SNN inputs)."""

    def __init__(self, n_classes: int = 5, n_rr: int = 0):
        super().__init__()
        self.n_rr = n_rr
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(32 + n_rr, n_classes)

    def forward(self, x: torch.Tensor, rr: torch.Tensor = None) -> torch.Tensor:
        h = self.net(x.unsqueeze(1)).squeeze(-1)  # [B, 32]
        if self.n_rr and rr is not None:
            h = torch.cat([h, rr], dim=1)
        return self.head(h)


class LSTMClassifier(nn.Module):
    """LSTM baseline on raw beats [B, L] (length-L, 1-feature seq), optionally with
    n_rr RR features concatenated before the head (fair match to the SNN inputs)."""

    def __init__(self, n_classes: int = 5, hidden: int = 64, n_rr: int = 0):
        super().__init__()
        self.n_rr = n_rr
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden + n_rr, n_classes)

    def forward(self, x: torch.Tensor, rr: torch.Tensor = None) -> torch.Tensor:
        out, _ = self.lstm(x.unsqueeze(-1))  # [B, L, 1]
        h = out[:, -1, :]
        if self.n_rr and rr is not None:
            h = torch.cat([h, rr], dim=1)
        return self.head(h)
