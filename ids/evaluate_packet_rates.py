
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from diagnosis_config import MAX_PUBLISHES_PER_WINDOW, RATE_WINDOW_SECONDS
from diagnosis_rules import rate_exceeded

INPUT = ROOT / "features" / "mqtt_traffic_features.csv"
OUTPUT = ROOT / "results" / "packet_rate_report.csv"

WINDOW_SECONDS = RATE_WINDOW_SECONDS
# Provisional: calibrated only against the single currently saved normal run.
RATE_THRESHOLD = MAX_PUBLISHES_PER_WINDOW

df = pd.read_csv(INPUT)

required = {"capture_file", "timestamp_epoch", "mqtt_message_type"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df["timestamp_epoch"] = pd.to_numeric(
    df["timestamp_epoch"], errors="coerce"
)
df["mqtt_message_type"] = df["mqtt_message_type"].astype(str)

# Keep only MQTT PUBLISH packets.
publishes = df[
    (df["mqtt_message_type"] == "3")
    & df["timestamp_epoch"].notna()
].copy()

publishes["excessive_rate_flag"] = False
publishes["rolling_window_count"] = 0

for capture, group in publishes.groupby("capture_file", sort=False):
    group = group.sort_values("timestamp_epoch")
    indices = group.index.tolist()
    times = group["timestamp_epoch"].tolist()

    left = 0
    for right, current_time in enumerate(times):
        while current_time - times[left] >= WINDOW_SECONDS:
            left += 1
        count = right - left + 1
        publishes.at[indices[right], "rolling_window_count"] = count
        publishes.at[indices[right], "excessive_rate_flag"] = rate_exceeded(
            count, RATE_THRESHOLD
        )

# Capture-level summary
summary_rows = []

for capture, group in publishes.groupby("capture_file", sort=False):
    times = group["timestamp_epoch"].sort_values()
    duration = float(times.iloc[-1] - times.iloc[0]) if len(times) > 1 else 0.0

    summary_rows.append({
        "capture_file": capture,
        "publish_packets": len(group),
        "duration_seconds": round(duration, 3),
        "average_rate_per_second": (
            round(len(group) / duration, 3)
            if duration > 0 else 0.0
        ),
        "window_seconds": WINDOW_SECONDS,
        "rate_threshold_publishes_per_window": RATE_THRESHOLD,
        "max_publishes_in_10_seconds": int(
            group["rolling_window_count"].max()
        ),
        "rate_flagged_packets": int(
            group["excessive_rate_flag"].sum()
        ),
    })

summary = pd.DataFrame(summary_rows)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUTPUT, index=False)

print("\nPacket-level rate evaluation:")
print(summary.to_string(index=False))
print(f"\nReport saved to: {OUTPUT}")
print(
    "\nRate definition: trailing interval (t-window, t], counting MQTT "
    "PUBLISH packets only. Threshold is provisional (20 per 10 seconds), "
    "based on one normal capture; flags are not proof of attack."
)
