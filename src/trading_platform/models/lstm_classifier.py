"""PyTorch LSTM classifier for 3-class direction labels."""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LSTMClassifier(nn.Module):
    """Multivariate LSTM stack with linear classification head."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 3,
    ) -> None:
        """Initialize layers.

        Args:
            input_size: Number of input features per time step.
            hidden_size: LSTM hidden units per layer.
            num_layers: Stacked LSTM depth.
            dropout: Dropout after LSTM (applied if num_layers > 1).
            num_classes: Output logits dimension.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for batch of sequences.

        Args:
            x: Tensor shape ``(batch, seq_len, input_size)``.

        Returns:
            Logits shape ``(batch, num_classes)``.
        """
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last)

    @staticmethod
    def predict_proba(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return class probabilities and max probability (confidence).

        Args:
            logits: Raw logits.

        Returns:
            Tuple ``(probs, confidence)`` each shape ``(batch,)`` for confidence
            as max prob; probs is full softmax.
        """
        probs = torch.softmax(logits, dim=-1)
        conf, _ = probs.max(dim=-1)
        return probs, conf
