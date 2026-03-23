from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from trading_platform.models.dataset import (
    LabelSpec,
    TimeSeriesSequenceDataset,
    build_arrays_from_frame,
    chronological_split_indices,
    fit_scaler_on_train,
    save_artifact_meta,
    transform_features,
)
from trading_platform.models.lstm_classifier import LSTMClassifier

logger = logging.getLogger(__name__)


def train_lstm_classifier(
    df,
    feature_cols: List[str],
    label_col: str,
    seq_len: int = 20,
    label_spec: Optional[LabelSpec] = None,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 64,
    epochs: int = 30,
    lr: float = 1e-3,
    device: Optional[str] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[LSTMClassifier, Path]:
    label_spec = label_spec or LabelSpec(horizon=1, hold_epsilon=0.002)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    artifacts_dir = artifacts_dir or Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    X_raw, y, _ = build_arrays_from_frame(df, feature_cols, label_col)
    if len(X_raw) < seq_len + 10:
        raise ValueError("Not enough rows after dropna for training")

    n_seq = len(X_raw) - seq_len + 1
    sl_train, sl_val, _ = chronological_split_indices(n_seq)
    n_train = sl_train.stop - sl_train.start
    last_train_row = seq_len - 1 + n_train - 1
    scaler = fit_scaler_on_train(X_raw[: last_train_row + 1])
    X_scaled = transform_features(X_raw, scaler)

    full_ds = TimeSeriesSequenceDataset(X_scaled, y, seq_len)
    train_ds = Subset(full_ds, range(sl_train.start, sl_train.stop))
    val_ds = Subset(full_ds, range(sl_val.start, sl_val.stop))

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

    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
        model.eval()
        vloss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(dev)
                yb = yb.to(dev)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                vloss += float(loss.item()) * len(xb)
                n_val += len(xb)
        vloss /= max(1, n_val)
        logger.info(
            "epoch %s train_loss=%.5f val_loss=%.5f",
            epoch + 1,
            total / max(1, len(train_ds)),
            vloss,
        )
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = artifacts_dir / "lstm_classifier.pt"
    torch.save(model.state_dict(), model_path)
    np.savez(
        artifacts_dir / "scaler.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        var=scaler.var_,
    )
    save_artifact_meta(artifacts_dir / "meta.json", feature_cols, seq_len, label_spec)
    logger.info("Saved model to %s", model_path)
    return model, artifacts_dir
