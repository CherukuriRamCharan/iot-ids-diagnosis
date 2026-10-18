"""Evaluate payload-value rules separately from packet-level rate rules."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from diagnosis_config import MAX_TEMPERATURE_C, MIN_TEMPERATURE_C
from diagnosis_rules import changed_fields, compare_message_pairs


INPUT = ROOT / "features" / "mqtt_payload_features.csv"
OUTPUT = ROOT / "results" / "rule_detection_report.csv"


def identify_scenario(path):
    name = str(path).lower()
    if "battery_draining" in name:
        return "excessive_communication_simulation"
    if "data_falsification" in name:
        return "data_falsification_simulation"
    if "mitm_simulation" in name:
        return "message_manipulation_simulation"
    if "replay" in name:
        return "replay_like_simulation"
    if "normal" in name:
        return "normal"
    return "unknown"


def main():
    df = pd.read_csv(INPUT, dtype={"sequence": "string"})
    required = {
        "capture_file", "timestamp_epoch", "mqtt_topic", "temperature",
        "motion", "payload_text", "decode_status",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    df["timestamp_epoch"] = pd.to_numeric(
        df["timestamp_epoch"], errors="coerce"
    )
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["motion"] = pd.to_numeric(df["motion"], errors="coerce")
    if "sequence" in df.columns:
        df["sequence"] = df["sequence"].fillna("").astype(str)
    df["range_flag"] = (
        df["temperature"].notna()
        & (
            (df["temperature"] < MIN_TEMPERATURE_C)
            | (df["temperature"] > MAX_TEMPERATURE_C)
        )
    )
    df["repeated_payload_evidence"] = False
    df["manipulation_flag"] = False
    df["manipulation_pairing_method"] = ""
    df["manipulation_changed_fields"] = ""
    df["manipulation_unmatched_rows"] = 0

    for capture, group in df.groupby("capture_file", sort=False):
        valid = group[group["decode_status"] == "decoded_json"]
        repeated = valid.duplicated(subset=["payload_text"], keep="first")
        df.loc[repeated[repeated].index, "repeated_payload_evidence"] = True

        decoded_group = group[group["decode_status"] == "decoded_json"]
        decoded_topics = decoded_group["mqtt_topic"].fillna("").astype(str)
        original = decoded_group[
            decoded_topics.str.endswith("/original")
        ].assign(_row_index=lambda frame: frame.index).to_dict("records")
        forwarded = decoded_group[
            decoded_topics.str.endswith("/forwarded")
        ].assign(_row_index=lambda frame: frame.index).to_dict("records")
        pairs, method, unmatched = compare_message_pairs(
            original, forwarded
        )
        candidate_rows = forwarded or original
        if candidate_rows:
            df.at[candidate_rows[0]["_row_index"], "manipulation_unmatched_rows"] = unmatched
            for candidate in candidate_rows:
                df.at[candidate["_row_index"], "manipulation_pairing_method"] = method
        for original_row, forwarded_row in pairs:
            changed = changed_fields(original_row, forwarded_row)
            idx = forwarded_row["_row_index"]
            if changed:
                df.at[idx, "manipulation_flag"] = True
                df.at[idx, "manipulation_changed_fields"] = ";".join(changed)

    df["expected_scenario"] = df["capture_file"].map(identify_scenario)
    df["rule_signals"] = df.apply(
        lambda row: ";".join(
            signal for condition, signal in (
                (bool(row["range_flag"]), "temperature_range_violation"),
                (bool(row["manipulation_flag"]), "paired_message_value_change"),
            ) if condition
        ) or "no_payload_rule_triggered",
        axis=1,
    )
    df.to_csv(OUTPUT, index=False)

    print("Payload rule summary (rate is evaluated separately):")
    print(
        df.groupby("expected_scenario", dropna=False).agg(
            payload_rows=("capture_file", "size"),
            decoded_json=("decode_status", lambda s: (s == "decoded_json").sum()),
            temperature_range_flags=("range_flag", "sum"),
            repeated_payload_evidence=("repeated_payload_evidence", "sum"),
            manipulation_flags=("manipulation_flag", "sum"),
        ).to_string()
    )
    print(f"Detailed report saved: {OUTPUT}")


if __name__ == "__main__":
    main()
