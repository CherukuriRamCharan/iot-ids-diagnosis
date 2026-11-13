from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
# predict_capture_attacks.py writes packet_attack_predictions.csv with
# columns: capture_file, label, predicted_class, benign, attack,
# total_packets, predicted_attack_percent.
# We derive the true capture label from the 'label' column.
summary_path = ROOT / "results" / "packet_attack_predictions.csv"
out_dir = ROOT / "results"
out_dir.mkdir(exist_ok=True)

if not summary_path.exists():
    raise FileNotFoundError(f"Missing file: {summary_path}")

df = pd.read_csv(summary_path)

required = {"label", "capture_file", "predicted_attack_percent"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

# Capture-level evaluation: attack if at least 50% of packets were predicted attack.
# 'label' is derived from the capture filename in the dataset pipeline.
df["true_label"] = df["label"].apply(
    lambda x: "benign" if str(x).lower() == "benign" else "attack"
)
df["predicted_label"] = df["predicted_attack_percent"].apply(
    lambda x: "attack" if float(x) >= 50 else "benign"
)

labels = ["benign", "attack"]
report = classification_report(
    df["true_label"], df["predicted_label"],
    labels=labels, output_dict=True, zero_division=0
)
matrix = confusion_matrix(
    df["true_label"], df["predicted_label"], labels=labels
)

result = {
    "evaluation_level": "capture-level, not packet-level",
    "number_of_captures": int(len(df)),
    "attack_threshold_percent": 50,
    "accuracy": float(accuracy_score(df["true_label"], df["predicted_label"])),
    "labels_order": labels,
    "confusion_matrix_rows_true_columns_predicted": matrix.tolist(),
    "classification_report": report,
    "limitation": (
        "This evaluates the saved model's capture-level predictions against "
        "filename-derived capture labels. It is not an independent test of "
        "the model and does not establish packet-level accuracy."
    )
}

print("CAPTURE-LEVEL FINAL EVALUATION")
print("=" * 45)
print(f"Captures evaluated: {len(df)}")
print(f"Accuracy: {result['accuracy']:.4f}")
print("\nConfusion matrix (rows=true, columns=predicted):")
print(pd.DataFrame(matrix, index=labels, columns=labels))
print("\nClassification report:")
print(classification_report(
    df["true_label"], df["predicted_label"],
    labels=labels, zero_division=0
))

csv_path = out_dir / "final_capture_evaluation.csv"
json_path = out_dir / "final_capture_evaluation.json"
df.to_csv(csv_path, index=False)
json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

print(f"\nSaved: {csv_path}")
print(f"Saved: {json_path}")
