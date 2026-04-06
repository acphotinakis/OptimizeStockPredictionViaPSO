"""
models/lstm/train.py
---------------------
Training script for LSTM classifier with early stopping,
checkpointing, and JSON experiment logging.

Usage
-----
    python -m models.lstm.train --config config.yaml --timeframe 1Min --horizon 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.lstm.model import LSTMClassifier, MultiHorizonLSTM
from preprocessing.feature_engineering import build_features
from preprocessing.labeling import build_labels
from preprocessing.scaling import FeatureScaler
from preprocessing.windowing import create_sequences, chronological_split
from preprocessing.feature_selection import remove_correlated_features
from data.ingestion import load_bars, load_config

logger = logging.getLogger(__name__)


def setup_device(use_gpu: bool) -> torch.device:
    """Select CUDA / MPS / CPU device."""
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    horizon_key: Optional[str] = None,
) -> float:
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).long()
        optimizer.zero_grad()
        if horizon_key:
            logits = model(X_batch)[horizon_key]
        else:
            logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    horizon_key: Optional[str] = None,
) -> Tuple[float, float]:
    """Evaluate model. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).long()
        if horizon_key:
            logits = model(X_batch)[horizon_key]
        else:
            logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(X_batch)
        preds = logits.argmax(dim=-1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def train_lstm(
    config: dict,
    timeframe: str = "1Min",
    horizon: int = 1,
    multi_horizon: bool = False,
) -> Dict:
    """
    Full LSTM training pipeline.

    Parameters
    ----------
    config : dict
    timeframe : str
    horizon : int
        Single horizon to train (ignored if multi_horizon=True).
    multi_horizon : bool

    Returns
    -------
    dict with training metrics and paths.
    """
    lstm_cfg = config["lstm"]
    device = setup_device(lstm_cfg.get("use_gpu", True))
    logger.info(f"Device: {device}")

    # --- Load data ---
    symbol = config["data"]["symbol"]
    df = load_bars(
        symbol, timeframe,
        config["data"]["raw_dir"],
        file_format=config["data"]["file_format"],
    )
    logger.info(f"Loaded {len(df):,} bars for {symbol} {timeframe}")

    # --- Features + Labels ---
    feat_df = build_features(df, config)
    label_df = build_labels(df, config)

    # Drop correlated features
    feat_df, _ = remove_correlated_features(
        feat_df,
        config["features"].get("correlation_threshold", 0.95),
    )

    # Align index
    combined = feat_df.join(label_df).dropna()
    feature_names = feat_df.columns.tolist()
    label_cols = [f"label_h{h}" for h in config["labeling"]["horizons"]]

    features_arr = combined[feature_names].values
    if multi_horizon:
        labels_arr = combined[label_cols].values.astype(np.float32)
    else:
        col = f"label_h{horizon}"
        labels_arr = combined[col].values.astype(np.float32)

    # --- Scale ---
    scaler = FeatureScaler(method="robust")
    features_arr = scaler.fit_transform(features_arr)

    # --- Windowing ---
    seq_len = lstm_cfg["sequence_length"]
    X, y, _ = create_sequences(features_arr, labels_arr, seq_len)

    # --- Split ---
    train_ratio = config["training"]["train_ratio"]
    val_ratio = config["training"]["val_ratio"]
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = chronological_split(
        X, y, train_ratio, val_ratio
    )

    def make_loader(X_, y_, shuffle=False):
        ds = TensorDataset(torch.from_numpy(X_), torch.from_numpy(y_))
        return DataLoader(ds, batch_size=lstm_cfg["batch_size"], shuffle=shuffle)

    train_loader = make_loader(X_tr, y_tr, shuffle=True)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(X_te, y_te)

    # --- Model ---
    input_size = X.shape[-1]
    if multi_horizon:
        model = MultiHorizonLSTM(
            input_size=input_size,
            hidden_size=lstm_cfg["hidden_size"],
            num_layers=lstm_cfg["num_layers"],
            horizons=config["labeling"]["horizons"],
            dropout=lstm_cfg["dropout"],
            bidirectional=lstm_cfg.get("bidirectional", False),
        ).to(device)
        horizon_key = f"h{config['labeling']['horizons'][0]}"
    else:
        model = LSTMClassifier(
            input_size=input_size,
            hidden_size=lstm_cfg["hidden_size"],
            num_layers=lstm_cfg["num_layers"],
            dropout=lstm_cfg["dropout"],
            bidirectional=lstm_cfg.get("bidirectional", False),
        ).to(device)
        horizon_key = None

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lstm_cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    # --- Training loop ---
    ckpt_dir = Path(lstm_cfg.get("checkpoint_dir", "checkpoints/lstm"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"lstm_{timeframe}_h{horizon}.pt"

    best_val_loss = float("inf")
    patience = lstm_cfg.get("early_stopping_patience", 10)
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, lstm_cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device, horizon_key)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, horizon_key)
        scheduler.step(val_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:03d} | tr_loss={tr_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.3f} | {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, str(ckpt_path))
            logger.info(f"  ✓ Checkpoint saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    # --- Final test evaluation ---
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, horizon_key)
    logger.info(f"Test | loss={test_loss:.4f} | acc={test_acc:.3f}")

    # --- Save scaler ---
    scaler_path = ckpt_dir / f"scaler_{timeframe}_h{horizon}.pkl"
    scaler.save(str(scaler_path))

    # --- Save feature names ---
    meta = {
        "feature_names": feature_names,
        "timeframe": timeframe,
        "horizon": horizon,
        "seq_len": seq_len,
        "input_size": input_size,
        "test_acc": test_acc,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    meta_path = ckpt_dir / f"meta_{timeframe}_h{horizon}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--timeframe", default="1Min")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--multi-horizon", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = train_lstm(cfg, args.timeframe, args.horizon, args.multi_horizon)
    print(f"\nTraining complete. Test accuracy: {result['test_acc']:.3f}")
