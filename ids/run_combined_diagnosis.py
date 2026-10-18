from pathlib import Path
from collections import Counter
import json

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "base_paper" / "all_packet_features.csv"
MODEL_PATH = ROOT / "models" / "base_paper_rf.joblib"
OUT_DIR = ROOT / "results"

OUT_DIR.mkdir(exist_ok=True)

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
features = bundle["features"]

# Detect available address/time fields without assuming they exist.
columns = pd.read_csv(DATA, nrows=0).columns.tolist()

def find_column(candidates):
    return next((c for c in candidates if c in columns), None)

src_col = find_column(["ip.src", "src_ip", "source_ip"])
dst_col = find_column(["ip.dst", "dst_ip", "destination_ip"])
time_col = find_column(["frame_time_epoch", "frame.time_epoch", "timestamp"])

stats = {}

print("Building combined ML + rule-based diagnosis...", flush=True)

for chunk in pd.read_csv(DATA, chunksize=100_000, low_memory=False):
    for col in features:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)

    chunk["ml_prediction"] = model.predict(chunk[features])
    chunk["is_attack"] = (chunk["ml_prediction"] == "attack").astype(int)

    if "tcp_syn" in chunk.columns:
        chunk["syn_flag"] = (chunk["tcp_syn"] > 0).astype(int)
    else:
        chunk["syn_flag"] = 0

    if "tcp_reset" in chunk.columns:
        chunk["reset_flag"] = (chunk["tcp_reset"] > 0).astype(int)
    else:
        chunk["reset_flag"] = 0

    for capture, group in chunk.groupby("capture_file", sort=False):
        label = str(group["label"].iloc[0])
        item = stats.setdefault(capture, {
            "capture_file": capture,
            "dataset_label": label,
            "total_packets": 0,
            "ml_attack_packets": 0,
            "ml_benign_packets": 0,
            "tcp_syn_packets": 0,
            "tcp_reset_packets": 0,
            "source_ips": Counter(),
            "destination_ips": Counter(),
            "min_time": None,
            "max_time": None,
        })

        item["total_packets"] += len(group)
        item["ml_attack_packets"] += int(group["is_attack"].sum())
        item["ml_benign_packets"] += int((group["is_attack"] == 0).sum())
        item["tcp_syn_packets"] += int(group["syn_flag"].sum())
        item["tcp_reset_packets"] += int(group["reset_flag"].sum())

        if src_col:
            item["source_ips"].update(
                group[src_col].dropna().astype(str).value_counts().to_dict()
            )
        if dst_col:
            item["destination_ips"].update(
                group[dst_col].dropna().astype(str).value_counts().to_dict()
            )
        if time_col:
            times = pd.to_numeric(group[time_col], errors="coerce").dropna()
            if not times.empty:
                low, high = float(times.min()), float(times.max())
                item["min_time"] = low if item["min_time"] is None else min(item["min_time"], low)
                item["max_time"] = high if item["max_time"] is None else max(item["max_time"], high)

rows = []

for item in stats.values():
    total = item["total_packets"]
    attack_count = item["ml_attack_packets"]
    attack_pct = 100 * attack_count / max(total, 1)
    syn_pct = 100 * item["tcp_syn_packets"] / max(total, 1)

    duration = None
    pps = None
    if item["min_time"] is not None and item["max_time"] is not None:
        duration = max(item["max_time"] - item["min_time"], 0)
        if duration > 0:
            pps = round(total / duration, 2)

    # Rules are evidence indicators, not definitive attack attribution.
    evidence = []
    if attack_pct >= 50:
        evidence.append("ML majority predicts attack")
    if syn_pct >= 20:
        evidence.append("high TCP SYN proportion")
    if pps is not None and pps > 100:
        evidence.append("high average packet rate")

    label = item["dataset_label"]
    if label == "benign":
        diagnosis = "benign reference capture; review ML flags"
    else:
        diagnosis = f"dataset-labelled {label} capture; inspect evidence"

    rows.append({
        "capture_file": item["capture_file"],
        "dataset_label": label,
        "diagnosis": diagnosis,
        "total_packets": total,
        "ml_attack_packets": attack_count,
        "ml_benign_packets": item["ml_benign_packets"],
        "ml_attack_percent": round(attack_pct, 2),
        "tcp_syn_packets": item["tcp_syn_packets"],
        "tcp_syn_percent": round(syn_pct, 2),
        "tcp_reset_packets": item["tcp_reset_packets"],
        "duration_seconds": round(duration, 3) if duration is not None else None,
        "average_packets_per_second": pps,
        "rule_evidence": "; ".join(evidence) if evidence else "no configured rule threshold triggered",
        "top_source_ip_observed": item["source_ips"].most_common(1)[0][0] if item["source_ips"] else None,
        "top_destination_ip_observed": item["destination_ips"].most_common(1)[0][0] if item["destination_ips"] else None,
        "localization_status": (
            "observed network address only; device/attacker identity unconfirmed"
            if src_col or dst_col else
            "unavailable: no source/destination IP columns in extracted data"
        ),
    })

result = pd.DataFrame(rows).sort_values(["dataset_label", "capture_file"])
csv_path = OUT_DIR / "combined_attack_diagnosis.csv"
json_path = OUT_DIR / "combined_attack_diagnosis.json"

result.to_csv(csv_path, index=False)
json_path.write_text(
    json.dumps(rows, indent=2, default=str),
    encoding="utf-8"
)

print("\nCapture diagnosis summary:")
print(result[[
    "capture_file", "dataset_label", "total_packets",
    "ml_attack_percent", "rule_evidence", "localization_status"
]].to_string(index=False))

print("\nCaptures:", len(result))
print("Packets accounted for:", int(result["total_packets"].sum()))
print("Expected packets: 2985994")
print("Counts match:", int(result["total_packets"].sum()) == 2985994)
print("CSV:", csv_path)
print("JSON:", json_path)
