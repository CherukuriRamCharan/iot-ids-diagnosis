"""
5-class Random Forest baseline with IAT and rolling-rate features.

Classes: benign, dos, mirai, mitm, scan
  - derived from capture filename (same as existing pipeline)

Splitting:
  - Attack captures: held out whole captures via GroupShuffleSplit (25% test).
  - Benign: temporal split (70/30 chronological) within the one benign capture.

Features used:
  packet_length, tcp_src_port, tcp_dst_port, udp_src_port, udp_dst_port,
  tcp_syn, tcp_ack, tcp_reset, udp_length,
  iat_seconds, rolling_10s_pkt_rate

Output:
  models/rf_5class.joblib
  results/rf_5class_evaluation.json

Run from any directory:
  .venv\\Scripts\\python.exe ids\\train_rf_5class.py

Limitation:
  Benign test set is from the same capture as benign training set (temporal
  proxy only).  Attack class labels are inferred from filenames.  This is a
  baseline — do not present as independent generalization evidence.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

ROOT       = Path(__file__).resolve().parent.parent
IAT_CSV    = ROOT / "data" / "base_paper" / "all_packet_features_iat.csv"
PLAIN_CSV  = ROOT / "data" / "base_paper" / "all_packet_features.csv"
MODEL_DIR  = ROOT / "models"
RESULT_DIR = ROOT / "results"

NUMERIC_COLS = [
    "packet_length", "tcp_src_port", "tcp_dst_port",
    "udp_src_port",  "udp_dst_port",
    "tcp_syn",       "tcp_ack",      "tcp_reset",
    "udp_length",
]

IAT_COLS = ["iat_seconds", "rolling_10s_pkt_rate"]

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
if IAT_CSV.exists():
    print(f"Loading enriched dataset with IAT features: {IAT_CSV}")
    DATA = IAT_CSV
    FEATURES = NUMERIC_COLS + IAT_COLS
else:
    print(
        f"WARNING: {IAT_CSV} not found. "
        "Run data/base_paper/add_iat_features.py first.\n"
        f"Falling back to plain CSV without IAT: {PLAIN_CSV}"
    )
    DATA = PLAIN_CSV
    FEATURES = NUMERIC_COLS

print("Loading dataset …", flush=True)
df = pd.read_csv(DATA, low_memory=False)
for col in FEATURES:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# 5-class label derived from filename prefix (already in 'label' column).
CLASS_MAP = {
    "benign": "benign",
    "dos":    "dos",
    "mirai":  "mirai",
    "mitm":   "mitm",
    "scan":   "scan",
}
df["target_5"] = df["label"].str.lower().map(CLASS_MAP).fillna("other")

print("Class distribution:")
print(df["target_5"].value_counts().to_string())

# ---------------------------------------------------------------------------
# Sampling per capture (cap memory; keep class balance reasonable)
# ---------------------------------------------------------------------------
MAX_PER_CAPTURE = 10_000
parts = []
for _, g in df.groupby("capture_file", sort=False):
    parts.append(g.sample(n=min(len(g), MAX_PER_CAPTURE), random_state=42))
sample = pd.concat(parts, ignore_index=True)

print(f"\nSampled rows: {len(sample):,}")
print("Sampled class distribution:")
print(sample["target_5"].value_counts().to_string())

# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------
benign  = sample[sample["target_5"] == "benign"].copy()
attacks = sample[sample["target_5"] != "benign"].copy()

# Hold out complete attack captures for test.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_a_idx, test_a_idx = next(
    splitter.split(attacks, attacks["target_5"], groups=attacks["capture_file"])
)
train_attacks = attacks.iloc[train_a_idx]
test_attacks  = attacks.iloc[test_a_idx]

# Temporal split for the single benign capture.
benign = benign.sort_values("timestamp") if "timestamp" in benign.columns else benign
cut         = int(len(benign) * 0.70)
train_benign = benign.iloc[:cut]
test_benign  = benign.iloc[cut:]

train = pd.concat([train_attacks, train_benign], ignore_index=True)
test  = pd.concat([test_attacks,  test_benign],  ignore_index=True)

# Ensure all expected feature columns exist.
for col in FEATURES:
    if col not in train.columns:
        train[col] = 0.0
        test[col]  = 0.0

X_train = train[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
y_train = train["target_5"]
X_test  = test[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
y_test  = test["target_5"]

print(f"\nTraining rows: {len(train):,}  |  Test rows: {len(test):,}")
print("Training labels:")
print(y_train.value_counts().to_string())
print("Test labels:")
print(y_test.value_counts().to_string())
print(f"Training attack captures: {train_attacks['capture_file'].nunique()}")
print(f"Test attack captures:     {test_attacks['capture_file'].nunique()}")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
print("\nTraining 5-class Random Forest …", flush=True)
model.fit(X_train, y_train)
pred = model.predict(X_test)

CLASSES = sorted(y_test.unique())

print("\nClassification report:")
print(classification_report(y_test, pred, labels=CLASSES, zero_division=0))
print("Confusion matrix (rows=true, columns=predicted):")
cm = confusion_matrix(y_test, pred, labels=CLASSES)
print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

joblib.dump({"model": model, "features": FEATURES, "classes": CLASSES},
            MODEL_DIR / "rf_5class.joblib")

summary = {
    "task": "5-class packet classification: benign / dos / mirai / mitm / scan",
    "features_used": FEATURES,
    "iat_features_included": IAT_CSV.exists(),
    "dataset_rows": int(len(df)),
    "capture_count": int(df["capture_file"].nunique()),
    "sampled_rows": int(len(sample)),
    "split_method": (
        "attack captures held out by capture (GroupShuffleSplit 75/25); "
        "benign split chronologically within the sole benign capture (70/30)"
    ),
    "training_label_counts":  y_train.value_counts().to_dict(),
    "testing_label_counts":   y_test.value_counts().to_dict(),
    "classes": CLASSES,
    "classification_report": classification_report(
        y_test, pred, labels=CLASSES, output_dict=True, zero_division=0
    ),
    "confusion_matrix_labels": CLASSES,
    "confusion_matrix": cm.tolist(),
    "limitations": [
        "Only one benign capture; benign test set is a same-capture temporal proxy.",
        "Labels inferred from filename; all packets per capture share the label.",
        "Packet-level features do not establish physical device localization.",
        "5-class model conflates sub-types (e.g. mirai-ackflooding vs mirai-hostbruteforce).",
        f"IAT features {'included' if IAT_CSV.exists() else 'NOT included (run add_iat_features.py first)'}.",
    ],
}
(RESULT_DIR / "rf_5class_evaluation.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(f"\nModel saved:      {MODEL_DIR / 'rf_5class.joblib'}")
print(f"Evaluation saved: {RESULT_DIR / 'rf_5class_evaluation.json'}")
