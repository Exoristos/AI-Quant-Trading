from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, Subset

from trading_platform.models.dataset import (
    TimeSeriesSequenceDataset,
    build_arrays_from_frame,
    fit_scaler_on_train,
    transform_features,
)
from trading_platform.models.lstm_classifier import LSTMClassifier

logger = logging.getLogger(__name__)


def walk_forward_lstm_metrics(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    seq_len: int,
    n_splits: int = 4,
    epochs_per_fold: int = 5,
    batch_size: int = 64,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    device: Optional[str] = None,
) -> pd.DataFrame:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X_raw, y, _ = build_arrays_from_frame(df, feature_cols, label_col)
    n_seq = len(X_raw) - seq_len + 1
    min_need = n_splits + seq_len + 30
    if n_seq < min_need:
        raise ValueError(f"Need more sequence samples (have {n_seq}, need ~{min_need})")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(np.arange(n_seq))):
        if len(train_idx) < 10 or len(val_idx) < 1:
            logger.warning("Skipping fold %s: train=%s val=%s", fold, len(train_idx), len(val_idx))
            continue
        last_train_end_row = seq_len - 1 + int(train_idx.max())
        scaler = fit_scaler_on_train(X_raw[: last_train_end_row + 1])
        X_scaled = transform_features(X_raw, scaler)
        full_ds = TimeSeriesSequenceDataset(X_scaled, y, seq_len)
        train_ds = Subset(full_ds, train_idx.tolist())
        val_ds = Subset(full_ds, val_idx.tolist())
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = LSTMClassifier(
            input_size=len(feature_cols),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=3,
        ).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs_per_fold):
            for xb, yb in train_loader:
                xb = xb.to(dev)
                yb = yb.to(dev)
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

        model.eval()
        total_loss = 0.0
        correct = 0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(dev)
                yb = yb.to(dev)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                total_loss += float(loss.item()) * len(xb)
                pred = logits.argmax(dim=-1)
                correct += int((pred == yb).sum().item())
                n_val += len(xb)
        val_loss = total_loss / max(1, n_val)
        val_acc = correct / max(1, n_val)
        logger.info("walk_forward fold=%s val_acc=%.4f val_loss=%.4f", fold, val_acc, val_loss)
        rows.append(
            {
                "fold": fold,
                "n_train_seq": len(train_idx),
                "n_val_seq": len(val_idx),
                "val_accuracy": val_acc,
                "val_loss": val_loss,
            }
        )
    return pd.DataFrame(rows)
