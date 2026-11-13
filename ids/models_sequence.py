"""PyTorch Neural Sequence Models for Stage 3C IoT Intrusion Detection.

Implements LSTMClassifier, BiLSTMClassifier, and GRUClassifier consuming sliding
windows of chronological packet feature vectors.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """Two-layer unidirectional LSTM classifier for sequential packet traffic."""

    def __init__(
        self,
        num_features: int = 33,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.layer_norm = nn.LayerNorm(num_features)
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        x_norm = self.layer_norm(x)
        out, _ = self.lstm(x_norm)
        last_step = out[:, -1, :]
        last_step = self.drop(last_step)
        return self.fc(last_step)


class BiLSTMClassifier(nn.Module):
    """Two-layer bidirectional LSTM classifier capturing past and future context within burst windows."""

    def __init__(
        self,
        num_features: int = 33,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.layer_norm = nn.LayerNorm(num_features)
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        x_norm = self.layer_norm(x)
        out, _ = self.lstm(x_norm)
        last_step = out[:, -1, :]
        last_step = self.drop(last_step)
        return self.fc(last_step)


class GRUClassifier(nn.Module):
    """Two-layer Gated Recurrent Unit (GRU) classifier for parameter-efficient temporal modeling."""

    def __init__(
        self,
        num_features: int = 33,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.layer_norm = nn.LayerNorm(num_features)
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, num_features).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        x_norm = self.layer_norm(x)
        out, _ = self.gru(x_norm)
        last_step = out[:, -1, :]
        last_step = self.drop(last_step)
        return self.fc(last_step)


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "lstm": LSTMClassifier,
    "bilstm": BiLSTMClassifier,
    "gru": GRUClassifier,
}


def build_model(
    model_name: str,
    *,
    num_features: int = 33,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_classes: int = 4,
    dropout: float = 0.2,
    **kwargs: Any,
) -> nn.Module:
    """Factory creating sequence classifier by model name."""
    key = model_name.strip().lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown sequence model: {model_name}. Supported: {list(MODEL_REGISTRY.keys())}"
        )
    cls = MODEL_REGISTRY[key]
    return cls(
        num_features=num_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        **kwargs,
    )

