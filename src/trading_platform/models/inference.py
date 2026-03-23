from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from trading_platform.models.dataset import LabelSpec, load_artifact_meta
from trading_platform.models.lstm_classifier import LSTMClassifier

logger = logging.getLogger(__name__)

CLASS_TO_SIGNAL = {0: -1, 1: 0, 2: 1}


def load_trained_bundle(
    artifacts_dir: Path,
    device: Optional[str] = None,
) -> Tuple[LSTMClassifier, StandardScaler, List[str], int, LabelSpec]:
    """Load weights, scaler, and metadata from disk.

    Args:
        artifacts_dir: Directory containing ``lstm_classifier.pt``, ``scaler.npz``, ``meta.json``.
        device: Torch device.

    Returns:
        model, scaler, feature_cols, seq_len, label_spec
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    meta_path = artifacts_dir / "meta.json"
    feature_cols, seq_len, label_spec = load_artifact_meta(meta_path)
    npz = np.load(artifacts_dir / "scaler.npz")
    scaler = StandardScaler()
    scaler.mean_ = npz["mean"]
    scaler.scale_ = npz["scale"]
    scaler.var_ = npz["var"]
    scaler.n_features_in_ = len(feature_cols)

    model = LSTMClassifier(input_size=len(feature_cols)).to(dev)
    state = torch.load(artifacts_dir / "lstm_classifier.pt", map_location=dev)
    model.load_state_dict(state)
    model.eval()
    return model, scaler, feature_cols, seq_len, label_spec


def predict_signals(
    df: pd.DataFrame,
    artifacts_dir: Path,
    device: Optional[str] = None,
) -> pd.DataFrame:
    model, scaler, feature_cols, seq_len, _ = load_trained_bundle(artifacts_dir, device=device)
    dev = next(model.parameters()).device
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing feature columns: {missing}")

    X = df[feature_cols].to_numpy(dtype=np.float64)
    Xs = scaler.transform(X).astype(np.float32)
    n = len(df)
    pred_class = np.full(n, np.nan)
    confidence = np.full(n, np.nan)
    with torch.no_grad():
        for end in range(seq_len - 1, n):
            start = end - seq_len + 1
            window = Xs[start : end + 1]
            xt = torch.from_numpy(window).unsqueeze(0).to(dev)
            logits = model(xt)
            probs = torch.softmax(logits, dim=-1)
            conf, cls = probs.max(dim=-1)
            pred_class[end] = int(cls.item())
            confidence[end] = float(conf.item())
    out = pd.DataFrame(
        {
            "pred_class": pred_class,
            "confidence": confidence,
        },
        index=df.index,
    )
    out["signal"] = np.nan
    valid = out["pred_class"].notna()
    out.loc[valid, "signal"] = out.loc[valid, "pred_class"].astype(int).map(CLASS_TO_SIGNAL)
    logger.info("Inference complete rows=%s valid_preds=%s", n, int(valid.sum()))
    return out
