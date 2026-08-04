"""Model definitions shared by training and Streamlit dashboard."""
from __future__ import annotations

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """One-step LSTM classifier used as an exploratory deep-learning comparator."""

    def __init__(self, input_dim: int, hidden_dim: int = 16, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.lstm(x)
        final_step = sequence_output[:, -1, :]
        return self.fc(self.dropout(final_step)).squeeze(1)
