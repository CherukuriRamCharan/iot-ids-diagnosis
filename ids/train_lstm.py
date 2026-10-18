"""
LSTM-based Intrusion Detection for IoT traffic.

Methodology (aligned with Kaddoura et al. 2026, §6):
  - Features: packet_length, iat_seconds, rolling_10s_pkt_rate,
               tcp_syn, tcp_ack, tcp_reset, udp_length
  - Windowed sequences of N_STEPS packets (per capture, time-ordered)
  - Label = majority class within the window
  - Capture-level train/val/test split (no window crosses capture boundary)

Usage:
  .venv\\Scripts\\python.exe ids\\train_lstm.py            # train
  .venv\\Scripts\\python.exe ids\\train_lstm.py --predict  # predict only

Output:
  models/lstm_ids.pt         (model checkpoint)
  results/lstm_evaluation.json

Limitations documented in output:
  - Only one benign capture available; class imbalance is handled via weights.
  - Labels inferred from filenames; window label = window majority class.
  - CPU-compatible; GPU used automatically if available.
  - With 42 captures, generalisation cannot be strongly claimed.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT       = Path(__file__).resolve().parent.parent
IAT_CSV    = ROOT / "data" / "base_paper" / "all_packet_features_iat.csv"
PLAIN_CSV  = ROOT / "data" / "base_paper" / "all_packet_features.csv"
MODEL_PATH = ROOT / "models" / "lstm_ids.pt"
RESULT_DIR = ROOT / "results"

# ---------------------------------------------------------------------------
# Hyper-parameters (document in output JSON)
# ---------------------------------------------------------------------------
N_STEPS    = 20        # packets per sequence window
N_HIDDEN   = 64        # LSTM hidden size
N_LAYERS   = 2         # stacked LSTM layers
DROPOUT    = 0.3
EPOCHS     = 30
BATCH_SIZE = 256
LR         = 1e-3
MAX_PER_CAPTURE = 6_000   # packets sampled per capture to cap memory

FEATURES_IAT   = [
    "packet_length", "iat_seconds", "rolling_10s_pkt_rate",
    "tcp_syn", "tcp_ack", "tcp_reset", "udp_length",
]
FEATURES_PLAIN = [
    "packet_length", "tcp_syn", "tcp_ack", "tcp_reset", "udp_length",
]

CLASS_MAP = {
    "benign": "benign",
    "dos":    "dos",
    "mirai":  "mirai",
    "mitm":   "mitm",
    "scan":   "scan",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    if IAT_CSV.exists():
        print(f"Loading IAT-enriched dataset: {IAT_CSV}")
        df = pd.read_csv(IAT_CSV, low_memory=False)
        features = FEATURES_IAT
    else:
        print(
            f"WARNING: {IAT_CSV} not found. "
            "Run data/base_paper/add_iat_features.py first.\n"
            f"Falling back to plain CSV (no IAT): {PLAIN_CSV}"
        )
        df = pd.read_csv(PLAIN_CSV, low_memory=False)
        features = FEATURES_PLAIN

    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    df["label_5"] = df["label"].str.lower().map(CLASS_MAP).fillna("other")
    df = df[df["label_5"] != "other"]   # drop unknown classes

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0.0)
        df = df.sort_values(["capture_file", "timestamp"])
    else:
        df = df.sort_values("capture_file")

    return df, features


# ---------------------------------------------------------------------------
# Sequence generation
# ---------------------------------------------------------------------------

def build_sequences(df, features, n_steps, max_per_capture):
    """
    Slide a window of n_steps packets over each capture.
    Label = majority class in the window.
    Captures with fewer than n_steps packets are skipped.
    """
    X_list, y_list, caps = [], [], []

    for cap, grp in df.groupby("capture_file", sort=False):
        grp = grp.reset_index(drop=True)
        if len(grp) < n_steps:
            continue

        # Sample to limit memory; preserve time order.
        if len(grp) > max_per_capture:
            keep = sorted(random.sample(range(len(grp)), max_per_capture))
            grp  = grp.iloc[keep].reset_index(drop=True)

        vals   = grp[features].values.astype(np.float32)
        labels = grp["label_5"].values

        for i in range(len(grp) - n_steps + 1):
            window_x = vals[i : i + n_steps]            # (n_steps, n_feat)
            window_y = labels[i : i + n_steps]
            maj_label = max(set(window_y), key=list(window_y).count)
            X_list.append(window_x)
            y_list.append(maj_label)
            caps.append(cap)

    if not X_list:
        raise ValueError("No sequences generated. Check dataset or n_steps.")

    return np.array(X_list, dtype=np.float32), np.array(y_list), np.array(caps)


# ---------------------------------------------------------------------------
# Train/val/test split by capture
# ---------------------------------------------------------------------------

def split_by_capture(X, y, caps, classes, train_ratio=0.70, val_ratio=0.15):
    """
    Split sequences by capture file so no capture appears in two splits.
    Stratification is approximate (small dataset).
    """
    unique_caps = list(dict.fromkeys(caps))   # preserve order, unique
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_caps)

    n      = len(unique_caps)
    n_train = max(1, int(n * train_ratio))
    n_val   = max(1, int(n * val_ratio))

    train_caps = set(unique_caps[:n_train])
    val_caps   = set(unique_caps[n_train : n_train + n_val])
    test_caps  = set(unique_caps[n_train + n_val:])

    if not test_caps:
        # Too few captures: move last val capture to test.
        moved = unique_caps[n_train + n_val - 1]
        val_caps.discard(moved)
        test_caps.add(moved)

    idx_train = np.where(np.isin(caps, list(train_caps)))[0]
    idx_val   = np.where(np.isin(caps, list(val_caps)))[0]
    idx_test  = np.where(np.isin(caps, list(test_caps)))[0]

    return (
        X[idx_train], y[idx_train],
        X[idx_val],   y[idx_val],
        X[idx_test],  y[idx_test],
        sorted(train_caps), sorted(val_caps), sorted(test_caps),
    )


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, X, y_enc):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_enc, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# LSTM model
# ---------------------------------------------------------------------------

class LSTMClassifier(nn.Module):
    def __init__(self, n_features, n_hidden, n_layers, n_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)
        out     = self.dropout(out[:, -1, :])   # last time step
        return self.fc(out)


# ---------------------------------------------------------------------------
# Training loop
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

    # --- Load ---
    df, features = load_data()
    n_feat = len(features)
    print(f"Features ({n_feat}): {features}")
    print(f"Total packets after cleaning: {len(df):,}")

    # --- Sequences ---
    print(f"\nBuilding sequences (window={N_STEPS}) …")
    X, y_str, caps = build_sequences(df, features, N_STEPS, MAX_PER_CAPTURE)
    print(f"Total sequences: {len(X):,}")
    unique, counts = np.unique(y_str, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {cls}: {cnt:,}")

    # --- Encode labels ---
    le = LabelEncoder()
    le.fit(sorted(CLASS_MAP.values()))
    y_enc = le.transform(y_str)
    classes = list(le.classes_)

    # --- Normalise features ---
    # Fit scaler on training data only (after split).
    (
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        train_caps, val_caps, test_caps,
    ) = split_by_capture(X, y_enc, caps, classes)

    print(f"\nTrain sequences: {len(X_train):,}  "
          f"Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"Train captures ({len(train_caps)}): {sorted(train_caps)[:3]} …")
    print(f"Val   captures ({len(val_caps)}):   {sorted(val_caps)[:3]} …")
    print(f"Test  captures ({len(test_caps)}):  {sorted(test_caps)}")

    if predict_only:
        if not MODEL_PATH.exists():
            raise SystemExit(f"No saved model at {MODEL_PATH}. Train first.")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        scaler_mean = np.array(checkpoint["scaler_mean"])
        scaler_std  = np.array(checkpoint["scaler_std"])
    else:
        # Fit scaler on train sequences (flatten to 2-D, fit, reshape back).
        flat_train = X_train.reshape(-1, n_feat)
        scaler_mean = flat_train.mean(axis=0)
        scaler_std  = flat_train.std(axis=0) + 1e-8
        scaler_mean = scaler_mean.astype(np.float32)
        scaler_std  = scaler_std.astype(np.float32)

    X_train = ((X_train - scaler_mean) / scaler_std).astype(np.float32)
    X_val   = ((X_val   - scaler_mean) / scaler_std).astype(np.float32)
    X_test  = ((X_test  - scaler_mean) / scaler_std).astype(np.float32)

    # --- Class weights ---
    class_counts = np.bincount(y_train, minlength=len(classes))
    class_weights = torch.tensor(
        1.0 / (class_counts + 1e-6), dtype=torch.float32
    ).to(device)
    class_weights = class_weights / class_weights.sum()

    # --- DataLoaders ---
    train_ds = SequenceDataset(X_train, y_train)
    val_ds   = SequenceDataset(X_val,   y_val)
    test_ds  = SequenceDataset(X_test,  y_test)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader  = torch.utils.data.DataLoader(
        val_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # --- Model ---
    model = LSTMClassifier(n_feat, N_HIDDEN, N_LAYERS, len(classes), DROPOUT).to(device)
    print(f"\nModel parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if predict_only:
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )

        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        ROOT.joinpath("models").mkdir(exist_ok=True)

        print("\nTraining LSTM …")
        for epoch in range(1, EPOCHS + 1):
            t_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            v_loss, v_acc, _, _ = eval_epoch(model, val_loader, criterion, device)
            scheduler.step(v_loss)

            history["train_loss"].append(round(t_loss, 5))
            history["val_loss"].append(round(v_loss, 5))
            history["val_acc"].append(round(v_acc, 5))

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d}/{EPOCHS}  "
                      f"train_loss={t_loss:.4f}  "
                      f"val_loss={v_loss:.4f}  "
                      f"val_acc={v_acc:.4f}")

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state":  model.state_dict(),
                        "scaler_mean":  scaler_mean.tolist(),
                        "scaler_std":   scaler_std.tolist(),
                        "features":     features,
                        "classes":      classes,
                        "n_hidden":     N_HIDDEN,
                        "n_layers":     N_LAYERS,
                        "dropout":      DROPOUT,
                        "n_steps":      N_STEPS,
                    },
                    MODEL_PATH,
                )

        print(f"\nBest val_loss: {best_val_loss:.4f}  (saved to {MODEL_PATH})")

        # Load best weights for test evaluation.
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])

    # --- Test evaluation ---
    _, _, pred_enc, true_enc = eval_epoch(
        model, test_loader,
        nn.CrossEntropyLoss(weight=class_weights),
        device,
    )
    pred_labels = le.inverse_transform(pred_enc)
    true_labels = le.inverse_transform(true_enc)

    print("\n=== LSTM TEST RESULTS ===")
    print(classification_report(true_labels, pred_labels,
                                labels=classes, zero_division=0))
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    print("Confusion matrix (rows=true, columns=predicted):")
    print(pd.DataFrame(cm, index=classes, columns=classes).to_string())

    RESULT_DIR.mkdir(exist_ok=True)
    result = {
        "model": "LSTM",
        "task":  "5-class sequence-level classification",
        "features": features,
        "iat_features_used": IAT_CSV.exists(),
        "n_steps":     N_STEPS,
        "n_hidden":    N_HIDDEN,
        "n_layers":    N_LAYERS,
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
            "Only one benign capture; benign class is underrepresented.",
            "Window label is assigned by majority vote within N_STEPS packets.",
            "With 42 captures, capture-level generalisation cannot be strongly claimed.",
            f"IAT features {'included' if IAT_CSV.exists() else 'NOT included — run add_iat_features.py first'}.",
            "Simulation attacks (replay, falsification) not in this dataset.",
        ],
    }
    (RESULT_DIR / "lstm_evaluation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"\nSaved: {RESULT_DIR / 'lstm_evaluation.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM IDS training/prediction")
    parser.add_argument(
        "--predict", action="store_true",
        help="Skip training; load saved model and evaluate on test split"
    )
    args = parser.parse_args()
    main(predict_only=args.predict)
