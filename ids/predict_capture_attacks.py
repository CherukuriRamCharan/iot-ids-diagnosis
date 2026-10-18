from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "base_paper" / "all_packet_features.csv"
MODEL_PATH = ROOT / "models" / "base_paper_rf.joblib"
OUT = ROOT / "results" / "packet_attack_predictions.csv"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

print("Loading packet features...", flush=True)
df = pd.read_csv(DATA, low_memory=False)

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Predict in batches to limit memory usage.
predictions = []
batch_size = 100_000

for start in range(0, len(df), batch_size):
    batch = df.iloc[start:start + batch_size]
    predictions.extend(model.predict(batch[features]))
    print(f"Predicted {min(start + batch_size, len(df))}/{len(df)} packets",
          flush=True)

df["predicted_class"] = predictions

summary = (
    df.groupby(["capture_file", "label", "predicted_class"])
      .size()
      .unstack(fill_value=0)
      .reset_index()
)

for col in ["benign", "attack"]:
    if col not in summary.columns:
        summary[col] = 0

summary["total_packets"] = summary["benign"] + summary["attack"]
summary["predicted_attack_percent"] = (
    100 * summary["attack"] / summary["total_packets"].clip(lower=1)
).round(2)

OUT.parent.mkdir(exist_ok=True)
summary.to_csv(OUT, index=False)

print("\nPrediction summary:")
print(summary.to_string(index=False))
print("\nSaved:", OUT)
