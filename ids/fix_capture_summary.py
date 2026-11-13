from pathlib import Path
import joblib
import pandas as pd

root = Path(__file__).resolve().parent.parent
source = root / "data" / "base_paper" / "all_packet_features.csv"
model_file = root / "models" / "base_paper_rf.joblib"
output = root / "results" / "capture_diagnosis_summary.csv"

bundle = joblib.load(model_file)
model = bundle["model"]
features = bundle["features"]

# Each capture has a fixed label. Aggregate numeric counters only.
counts = {}
for chunk in pd.read_csv(source, chunksize=100_000, low_memory=False):
    for col in features:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)

    predicted = model.predict(chunk[features])
    chunk["predicted_attack"] = (predicted == "attack").astype("int64")
    chunk["predicted_benign"] = (predicted == "benign").astype("int64")

    part = chunk.groupby(["capture_file", "label"]).agg(
        total_packets=("predicted_attack", "size"),
        predicted_attack=("predicted_attack", "sum"),
        predicted_benign=("predicted_benign", "sum"),
    )

    for key, row in part.iterrows():
        if key not in counts:
            counts[key] = {
                "total_packets": 0,
                "predicted_attack": 0,
                "predicted_benign": 0,
            }
        for col in counts[key]:
            counts[key][col] += int(row[col])

rows = []
for (capture, label), values in counts.items():
    rows.append({
        "capture_file": capture,
        "true_capture_label": label,
        **values,
        "predicted_attack_percent": round(
            100 * values["predicted_attack"] / max(values["total_packets"], 1), 2
        ),
    })

result = pd.DataFrame(rows).sort_values(["true_capture_label", "capture_file"])
result.to_csv(output, index=False)

print(result.to_string(index=False))
print("\nValidation: total packet count =", int(result["total_packets"].sum()))
print("Expected packet count = 2985994")
print("Counts match:", int(result["total_packets"].sum()) == 2985994)
print("Saved:", output)
