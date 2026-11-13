"""
Transformer-based Intrusion Detection for IoT traffic.

Architecture: encoder-only Transformer (stack of self-attention layers)
  on windowed packet sequences — same data pipeline as train_lstm.py.

Same feature set, same capture-level train/val/test split, same scaler.
Saves results/transformer_evaluation.json and models/transformer_ids.pt.

Usage:
  .venv\\Scripts\\python.exe ids\\train_transformer.py
  .venv\\Scripts\\python.exe ids\\train_transformer.py --predict

Limitations:
  - Same data limitations as train_lstm.py.
  - Small dataset makes Transformer prone to overfitting vs LSTM; check val curves.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Re-use data utilities from train_lstm.py
# ---------------------------------------------------------------------------
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ids"))

from train_lstm import (
    IAT_CSV, PLAIN_CSV, FEATURES_IAT, FEATURES_PLAIN,
    CLASS_MAP, MAX_PER_CAPTURE, SEED,
    load_data, build_sequences, split_by_capture, SequenceDataset,
    train_epoch as _train_epoch_lstm,   # we define our own below
    eval_epoch as _eval_epoch_lstm,
)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
N_STEPS    = 20
D_MODEL    = 64      # embedding dimension
N_HEADS    = 4
N_LAYERS   = 2
DIM_FF     = 128
DROPOUT    = 0.2
EPOCHS     = 30
BATCH_SIZE = 256
LR         = 5e-4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_PATH = ROOT / "models" / "transformer_ids.pt"
RESULT_DIR = ROOT / "results"


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Classifier
# ---------------------------------------------------------------------------

class TransformerClassifier(nn.Module):
    def __init__(self, n_features, d_model, n_heads, n_layers,
                 dim_ff, n_classes, dropout):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool    = nn.AdaptiveAvgPool1d(1)   # global average pooling
        self.fc      = nn.Linear(d_model, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)        # (batch, seq_len, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)           # (batch, seq_len, d_model)
        x = x.transpose(1, 2)        # (batch, d_model, seq_len)
        x = self.pool(x).squeeze(-1) # (batch, d_model)
        x = self.dropout(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Training / evaluation loops (same as LSTM, duplicated for clarity)
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits  = model(xb)
            loss    = criterion(logits, yb)
            total_loss += loss.item() * len(xb)
            all_pred.extend(logits.argmax(dim=1).cpu().tolist())
            all_true.extend(yb.cpu().tolist())
    n = len(loader.dataset) if len(loader.dataset) > 0 else 1
    avg_loss = total_loss / n
    accuracy = sum(p == t for p, t in zip(all_pred, all_true)) / max(len(all_true), 1)
    return avg_loss, accuracy, all_pred, all_true


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(predict_only=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df, features = load_data()
    n_feat = len(features)
    print(f"Features ({n_feat}): {features}")

    print(f"Building sequences (window={N_STEPS}) …")
    X, y_str, caps = build_sequences(df, features, N_STEPS, MAX_PER_CAPTURE)
    print(f"Total sequences: {len(X):,}")

    le = LabelEncoder()
    le.fit(sorted(CLASS_MAP.values()))
    y_enc   = le.transform(y_str)
    classes = list(le.classes_)

    (
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        train_caps, val_caps, test_caps,
    ) = split_by_capture(X, y_enc, caps, classes)

    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    if predict_only:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        scaler_mean = np.array(checkpoint["scaler_mean"], dtype=np.float32)
        scaler_std  = np.array(checkpoint["scaler_std"],  dtype=np.float32)
    else:
        flat   = X_train.reshape(-1, n_feat)
        scaler_mean = flat.mean(axis=0).astype(np.float32)
        scaler_std  = (flat.std(axis=0) + 1e-8).astype(np.float32)

    X_train = ((X_train - scaler_mean) / scaler_std).astype(np.float32)
    X_val   = ((X_val   - scaler_mean) / scaler_std).astype(np.float32)
    X_test  = ((X_test  - scaler_mean) / scaler_std).astype(np.float32)

    class_counts  = np.bincount(y_train, minlength=len(classes))
    class_weights = torch.tensor(1.0 / (class_counts + 1e-6), dtype=torch.float32).to(device)
    class_weights = class_weights / class_weights.sum()

    train_loader = torch.utils.data.DataLoader(
        SequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        SequenceDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        SequenceDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = TransformerClassifier(
        n_feat, D_MODEL, N_HEADS, N_LAYERS, DIM_FF, len(classes), DROPOUT
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    if not predict_only:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        best_val_loss = float("inf")
        print("\nTraining Transformer …")
        for epoch in range(1, EPOCHS + 1):
            t_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            v_loss, v_acc, _, _ = eval_epoch(model, val_loader, criterion, device)
            scheduler.step(v_loss)

            history["train_loss"].append(round(t_loss, 5))
            history["val_loss"].append(round(v_loss, 5))
            history["val_acc"].append(round(v_acc, 5))

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                      f"train_loss={t_loss:.4f}  val_loss={v_loss:.4f}  val_acc={v_acc:.4f}")

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                ROOT.joinpath("models").mkdir(exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state":  model.state_dict(),
                        "scaler_mean":  scaler_mean.tolist(),
                        "scaler_std":   scaler_std.tolist(),
                        "features":     features,
                        "classes":      classes,
                        "d_model":      D_MODEL,
                        "n_heads":      N_HEADS,
                        "n_layers":     N_LAYERS,
                        "n_steps":      N_STEPS,
                    },
                    MODEL_PATH,
                )

        print(f"\nBest val_loss: {best_val_loss:.4f}  saved to {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])

    else:
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

    _, _, pred_enc, true_enc = eval_epoch(
        model, test_loader,
        nn.CrossEntropyLoss(weight=class_weights),
        device,
    )
    pred_labels = le.inverse_transform(pred_enc)
    true_labels = le.inverse_transform(true_enc)

    print("\n=== TRANSFORMER TEST RESULTS ===")
    print(classification_report(true_labels, pred_labels, labels=classes, zero_division=0))
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    print("Confusion matrix:")
    print(pd.DataFrame(cm, index=classes, columns=classes).to_string())

    RESULT_DIR.mkdir(exist_ok=True)
    result = {
        "model":       "Transformer",
        "task":        "5-class sequence-level classification",
        "features":    features,
        "iat_features_used": IAT_CSV.exists(),
        "n_steps":     N_STEPS,
        "d_model":     D_MODEL,
        "n_heads":     N_HEADS,
        "n_layers":    N_LAYERS,
        "dim_ff":      DIM_FF,
        "dropout":     DROPOUT,
        "epochs":      EPOCHS,
        "batch_size":  BATCH_SIZE,
        "lr":          LR,
        "train_captures": sorted(train_caps),
        "val_captures":   sorted(val_caps),
        "test_captures":  sorted(test_caps),
        "train_sequences": int(len(X_train)),
        "val_sequences":   int(len(X_val)),
        "test_sequences":  int(len(X_test)),
        "classes": classes,
        "classification_report": classification_report(
            true_labels, pred_labels,
            labels=classes, output_dict=True, zero_division=0
        ),
        "confusion_matrix_labels": classes,
        "confusion_matrix": cm.tolist(),
        "training_history": history if (not predict_only and 'history' in dir()) else "predict_only",
        "limitations": [
            "Only one benign capture; class imbalance may inflate attack-class metrics.",
            "Transformer may overfit on 42 captures compared to LSTM.",
            "Window label assigned by majority vote within window.",
            f"IAT features {'included' if IAT_CSV.exists() else 'NOT included'}.",
        ],
    }
    (RESULT_DIR / "transformer_evaluation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"Saved: {RESULT_DIR / 'transformer_evaluation.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer IDS training/prediction")
    parser.add_argument("--predict", action="store_true", help="Predict only, skip training")
    args = parser.parse_args()
    main(predict_only=args.predict)
