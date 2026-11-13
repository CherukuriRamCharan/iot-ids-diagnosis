"""Combine packet, payload, and scenario metadata without conflating them."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from diagnosis_config import (
    MAX_PUBLISHES_PER_WINDOW,
    MAX_TEMPERATURE_C,
    MIN_TEMPERATURE_C,
    RATE_WINDOW_SECONDS,
)
from diagnosis_rules import diagnosis_signals

PAYLOAD_FILE = ROOT / "features" / "mqtt_payload_features.csv"
RULE_FILE = ROOT / "results" / "rule_detection_report.csv"
RATE_FILE = ROOT / "results" / "packet_rate_report.csv"
TRAFFIC_FILE = ROOT / "results" / "mqtt_traffic_summary.csv"
OUTPUT_FILE = ROOT / "results" / "combined_diagnosis_report.csv"


def _scenario(path):
    name = str(path).lower()
    for marker, label in (
        ("battery_draining", "excessive_communication_simulation"),
        ("data_falsification", "data_falsification_simulation"),
        ("mitm_simulation", "message_manipulation_simulation"),
        ("replay", "replay_like_simulation"),
        ("normal", "normal"),
    ):
        if marker in name:
            return label
    return "unknown"


def main():
    payload = pd.read_csv(PAYLOAD_FILE)
    rules = pd.read_csv(RULE_FILE)
    rates = pd.read_csv(RATE_FILE)
    traffic = pd.read_csv(TRAFFIC_FILE)

    for frame, name in (
        (payload, "payload"), (rules, "rules"), (rates, "rates"),
        (traffic, "traffic"),
    ):
        if "capture_file" not in frame:
            raise ValueError(f"{name} data has no capture_file column")

    if rates["capture_file"].duplicated().any():
        raise ValueError("Rate report has duplicate capture_file rows")
    if traffic["capture_file"].duplicated().any():
        raise ValueError("Traffic summary has duplicate capture_file rows")

    payload["temperature_numeric"] = pd.to_numeric(
        payload["temperature"], errors="coerce"
    )
    payload["motion_numeric"] = pd.to_numeric(
        payload["motion"], errors="coerce"
    )
    payload_summary = payload.groupby("capture_file", dropna=False).agg(
        traffic_label=("traffic_label", "first"),
        decoded_payload_count=("decode_status", lambda s: (s == "decoded_json").sum()),
        malformed_payload_count=("decode_status", lambda s: (s != "decoded_json").sum()),
        payload_count=("capture_file", "size"),
        temperature_min=("temperature_numeric", "min"),
        temperature_max=("temperature_numeric", "max"),
        motion_zero_rows=("motion_numeric", lambda s: (s == 0).sum()),
        motion_one_rows=("motion_numeric", lambda s: (s == 1).sum()),
        device_ids=("device_id", lambda s: ";".join(sorted(set(
            str(value) for value in s.dropna() if str(value).strip()
        )))),
        topics=("mqtt_topic", lambda s: ";".join(sorted(set(
            str(value) for value in s.dropna() if str(value).strip()
        )))),
    ).reset_index()

    rule_summary = rules.groupby("capture_file", dropna=False).agg(
        temperature_range_violation_rows=("range_flag", "sum"),
        repeated_payload_evidence_rows=("repeated_payload_evidence", "sum"),
        manipulation_mismatch_rows=("manipulation_flag", "sum"),
        manipulation_unmatched_rows=("manipulation_unmatched_rows", "max"),
        manipulation_pairing_methods=("manipulation_pairing_method",
                                      lambda s: ";".join(sorted(set(
                                          str(value) for value in s
                                          if str(value).strip()
                                      )))),
    ).reset_index()

    report = traffic.merge(
        payload_summary, on="capture_file", how="outer",
        validate="one_to_one", suffixes=("_traffic", "_payload"),
    )
    report = report.merge(
        rates, on="capture_file", how="outer", validate="one_to_one",
        suffixes=("", "_rate"),
    )
    report = report.merge(
        rule_summary, on="capture_file", how="left", validate="one_to_one",
    )
    if "traffic_label_payload" in report:
        traffic_labels = report["traffic_label_payload"]
    elif "traffic_label_traffic" in report:
        traffic_labels = report["traffic_label_traffic"]
    else:
        traffic_labels = pd.Series("unknown", index=report.index)
    report["traffic_label"] = traffic_labels.fillna("unknown")
    report["simulated_scenario"] = report["capture_file"].map(_scenario)

    for column in (
        "temperature_range_violation_rows", "manipulation_mismatch_rows",
        "rate_flagged_packets",
    ):
        report[column] = pd.to_numeric(report[column], errors="coerce").fillna(0)

    def signals(row):
        return diagnosis_signals(
            row["temperature_range_violation_rows"],
            row["rate_flagged_packets"],
            row["manipulation_mismatch_rows"],
        )

    report["detection_signals"] = report.apply(
        lambda row: ";".join(signals(row)) or "none", axis=1
    )
    report["diagnosis"] = report.apply(
        lambda row: (
            ";".join(signals(row))
            if signals(row)
            else "no_rule_signal"
        ), axis=1
    )
    report["attack_confirmed"] = False
    report["supporting_evidence"] = report.apply(
        lambda row: "; ".join(
            evidence for condition, evidence in (
                (row["temperature_range_violation_rows"] > 0,
                 f'{int(row["temperature_range_violation_rows"])} temperature value(s) outside illustrative limits [{MIN_TEMPERATURE_C}, {MAX_TEMPERATURE_C}] C'),
                (row["rate_flagged_packets"] > 0,
                 f'{int(row["rate_flagged_packets"])} PUBLISH packet(s) exceeded the provisional trailing-{RATE_WINDOW_SECONDS:g}-second limit of {MAX_PUBLISHES_PER_WINDOW}'),
                (row["manipulation_mismatch_rows"] > 0,
                 f'{int(row["manipulation_mismatch_rows"])} paired original/forwarded payload(s) differed'),
                (row.get("manipulation_unmatched_rows", 0) > 0,
                 f'{int(row.get("manipulation_unmatched_rows", 0))} original/forwarded payload row(s) could not be paired'),
                (row.get("repeated_payload_evidence_rows", 0) > 0,
                 f'{int(row.get("repeated_payload_evidence_rows", 0))} exact payload repeat(s); evidence only, not an attack signal'),
            ) if condition
        ) or "No configured payload/rate rule produced a signal.",
        axis=1,
    )
    report["uncertainty_and_limitations"] = (
        "Scenario name is simulation metadata, not an independently verified attack; "
        f"rate threshold ({MAX_PUBLISHES_PER_WINDOW} per {RATE_WINDOW_SECONDS:g} seconds) is provisional from one normal capture; repeated values alone "
        "are not treated as attacks; no physical source localization is supported."
    )

    columns = [
        "capture_file", "traffic_label", "simulated_scenario",
        "mqtt_packets", "mqtt_publish_packets", "payload_count",
        "decoded_payload_count", "malformed_payload_count",
        "temperature_min", "temperature_max", "motion_zero_rows",
        "motion_one_rows", "device_ids", "topics",
        "capture_duration_seconds", "publish_duration_seconds",
        "publish_packets_per_second",
        "average_rate_per_second", "window_seconds",
        "rate_threshold_publishes_per_window",
        "max_publishes_in_10_seconds", "rate_flagged_packets",
        "temperature_range_violation_rows",
        "repeated_payload_evidence_rows", "manipulation_mismatch_rows",
        "manipulation_unmatched_rows", "manipulation_pairing_methods",
        "detection_signals", "diagnosis",
        "attack_confirmed", "supporting_evidence",
        "uncertainty_and_limitations",
    ]
    report = report.reindex(columns=columns)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_FILE, index=False)
    print("Combined per-capture diagnosis report:")
    print(report.to_string(index=False))
    print(f"Saved report: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
