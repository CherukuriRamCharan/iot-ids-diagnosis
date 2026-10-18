import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "base_paper" / "all_packet_features.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

numeric_cols = [
    "packet_length", "tcp_src_port", "tcp_dst_port",
    "udp_src_port", "udp_dst_port", "tcp_syn", "tcp_ack",
    "tcp_reset", "udp_length",
]

print("Loading dataset...", flush=True)
df = pd.read_csv(DATA, low_memory=False)
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Sample each capture to limit memory and training time.
parts = []
for _, g in df.groupby("capture_file", sort=False):
    parts.append(g.sample(n=min(len(g), 12000), random_state=42))
sample = pd.concat(parts, ignore_index=True)
sample["target"] = np.where(sample["label"].eq("benign"), "benign", "attack")

benign = sample[sample["target"] == "benign"].copy()
attacks = sample[sample["target"] == "attack"].copy()

# Hold out whole attack captures.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_a_idx, test_a_idx = next(
    splitter.split(attacks, attacks["target"], groups=attacks["capture_file"])
)
train_attacks = attacks.iloc[train_a_idx]
test_attacks = attacks.iloc[test_a_idx]

# Same-capture temporal proxy: earlier benign packets train, later packets test.
benign = benign.sort_values("frame_time_epoch") if "frame_time_epoch" in benign.columns else benign
cut = int(len(benign) * 0.70)
train_benign = benign.iloc[:cut]
test_benign = benign.iloc[cut:]

train = pd.concat([train_attacks, train_benign], ignore_index=True)
test = pd.concat([test_attacks, test_benign], ignore_index=True)

X_train = train[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y_train = train["target"]
X_test = test[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y_test = test["target"]

print("Training rows:", len(train), "| Test rows:", len(test))
print("Training labels:", y_train.value_counts().to_dict())
print("Test labels:", y_test.value_counts().to_dict())
print("Training attack captures:", train_attacks["capture_file"].nunique())
print("Test attack captures:", test_attacks["capture_file"].nunique())

model = RandomForestClassifier(
    n_estimators=120, max_depth=18, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1
)
print("Training Random Forest...", flush=True)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nClassification report:")
print(classification_report(y_test, pred, labels=["benign", "attack"], zero_division=0))
print("Confusion matrix (rows=true, columns=predicted; benign, attack):")
print(confusion_matrix(y_test, pred, labels=["benign", "attack"]))

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
joblib.dump({"model": model, "features": numeric_cols},
            MODEL_DIR / "base_paper_rf.joblib")

summary = {
    "task": "binary packet classification: benign vs attack",
    "dataset_rows": int(len(df)),
    "capture_count": int(df["capture_file"].nunique()),
    "sampled_rows": int(len(sample)),
    "split_method": "attack captures held out by capture; benign split within the sole benign capture",
    "training_label_counts": y_train.value_counts().to_dict(),
    "testing_label_counts": y_test.value_counts().to_dict(),
    "classification_report": classification_report(
        y_test, pred, labels=["benign", "attack"],
        output_dict=True, zero_division=0
    ),
    "confusion_matrix_labels": ["benign", "attack"],
    "confusion_matrix": confusion_matrix(
        y_test, pred, labels=["benign", "attack"]
    ).tolist(),
    "features": numeric_cols,
    "limitations": [
        "Only one benign capture is available; benign testing is a same-capture temporal proxy.",
        "Labels are inferred from capture filenames and apply to every packet in each capture.",
        "Packet-level features do not establish physical IoT device localization.",
        "Results are a baseline and require validation on additional benign and attack captures."
    ]
}
(RESULT_DIR / "base_paper_model_evaluation.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print("\nUpdated model and evaluation saved.")
