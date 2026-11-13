"""Unified Four-Sensor IoT Dataset Generation Pipeline.

Generates a leak-free, reproducible multi-sensor dataset across 24 simulated
sessions using the UnifiedSimulator and AttackInjector.

Key Principles:
1. Four Sensors: Motion, Temperature, Ultrasonic, RFID.
2. Seven Classes: Normal + 6 Attacks (DoS, Replay, MITM, DataFalsification, Delay, IdentitySpoofing).
3. Session-Level Splitting: Train (sessions 0-13), Val (sessions 14-18), Test (sessions 19-23).
   Zero session overlap to guarantee no near-duplicate leakage between train and test.
4. Strict Feature/Label Quarantine:
   Observable features (X) NEVER contain ground-truth labels, internal TFSM states,
   output symbols, original payloads, or attack-injector metadata.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from simulation.attack_injector import AttackInjector, AttackSpec
from simulation.unified_simulator import UnifiedEvent, UnifiedSimulator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "datasets" / "generated" / "four_sensors"

OBSERVABLE_FEATURE_COLUMNS: tuple[str, ...] = (
    "inter_arrival",
    "payload_size",
    "qos",
    "seq_delta",
    "is_sensor_motion",
    "is_sensor_temp",
    "is_sensor_ultrasonic",
    "is_sensor_rfid",
    "is_event_publish",
    "is_event_alert",
    "is_event_heartbeat",
    "is_event_connect",
    "is_event_flood",
    "numeric_reading",
    "has_alert_flag",
    "has_tamper_flag",
)

GROUND_TRUTH_COLUMNS: tuple[str, ...] = (
    "is_attack",
    "attack_type",
    "attack_target",
    "fault_location",
)

METADATA_COLUMNS: tuple[str, ...] = (
    "session_id",
    "split",
    "timestamp",
    "sensor_id",
    "topic",
    "event_type",
    "sequence_number",
    "observed_state",
    "output_symbol",
)


def extract_features_from_event(
    event: UnifiedEvent,
    last_seq_by_sensor: Dict[str, int],
) -> Dict[str, Any]:
    """Extracts strictly observable network and payload features from an event."""
    sid = event.sensor_id
    ev_type = event.event_type

    # Sequence number delta
    prev_seq = last_seq_by_sensor.get(sid, event.sequence_number - 1)
    seq_delta = event.sequence_number - prev_seq
    last_seq_by_sensor[sid] = event.sequence_number

    # One-hot sensor indicators
    is_motion = 1.0 if sid == "motion" else 0.0
    is_temp = 1.0 if sid == "temperature" else 0.0
    is_ultra = 1.0 if sid == "ultrasonic" else 0.0
    is_rfid = 1.0 if sid == "rfid" else 0.0

    # One-hot event type indicators
    is_pub = 1.0 if ev_type == "PUBLISH" else 0.0
    is_alert = 1.0 if ev_type == "ALERT" else 0.0
    is_hb = 1.0 if ev_type == "HEARTBEAT" else 0.0
    is_conn = 1.0 if ev_type == "CONNECT" else 0.0
    is_flood = 1.0 if ev_type == "FLOOD" else 0.0

    # Numeric reading extraction
    payload = event.payload or {}
    if "temperature" in payload:
        num_read = float(payload["temperature"])
    elif "distance_cm" in payload:
        num_read = float(payload["distance_cm"])
    elif "motion" in payload:
        num_read = float(payload["motion"])
    elif "reader_gate" in payload:
        num_read = 1.0 if payload.get("status") == "authorized" else 0.0
    else:
        num_read = -1.0  # Sentinel for non-telemetry / flood frames

    has_alert = 1.0 if bool(payload.get("alert") or payload.get("obstacle") or payload.get("motion")) else 0.0
    has_tamper = 1.0 if "tampered_by" in payload else 0.0

    return {
        "inter_arrival": float(event.inter_arrival),
        "payload_size": float(event.payload_size),
        "qos": float(event.qos),
        "seq_delta": float(seq_delta),
        "is_sensor_motion": is_motion,
        "is_sensor_temp": is_temp,
        "is_sensor_ultrasonic": is_ultra,
        "is_sensor_rfid": is_rfid,
        "is_event_publish": is_pub,
        "is_event_alert": is_alert,
        "is_event_heartbeat": is_hb,
        "is_event_connect": is_conn,
        "is_event_flood": is_flood,
        "numeric_reading": round(num_read, 2),
        "has_alert_flag": has_alert,
        "has_tamper_flag": has_tamper,
    }


def define_session_scenarios() -> Dict[int, Tuple[str, List[AttackSpec]]]:
    """Defines 24 deterministic simulation sessions across train, val, and test."""
    # (split, list of attack specs)
    scenarios: Dict[int, Tuple[str, List[AttackSpec]]] = {}

    # --- Training Sessions (0 - 13: 14 sessions) ---
    scenarios[0] = ("train", [])  # Benign
    scenarios[1] = ("train", [AttackSpec("DoS", "ultrasonic", start_time=8.0, duration=3.0, intensity=1.5)])
    scenarios[2] = ("train", [AttackSpec("Replay", "motion", start_time=10.0, duration=6.0, params={"delay_offset": 4.0})])
    scenarios[3] = ("train", [AttackSpec("MITM", "temperature", start_time=7.0, duration=5.0)])
    scenarios[4] = ("train", [AttackSpec("DataFalsification", "temperature", start_time=12.0, duration=5.0)])
    scenarios[5] = ("train", [AttackSpec("Delay", "ultrasonic", start_time=6.0, duration=4.0, params={"delay_sec": 3.5})])
    scenarios[6] = ("train", [AttackSpec("IdentitySpoofing", "rfid", start_time=5.0, duration=15.0)])
    scenarios[7] = ("train", [AttackSpec("DoS", "motion", start_time=15.0, duration=3.0, intensity=1.2)])
    scenarios[8] = ("train", [AttackSpec("Replay", "temperature", start_time=14.0, duration=5.0, params={"delay_offset": 3.0})])
    scenarios[9] = ("train", [AttackSpec("MITM", "ultrasonic", start_time=9.0, duration=4.0)])
    scenarios[10] = ("train", [AttackSpec("DataFalsification", "motion", start_time=8.0, duration=4.0)])
    scenarios[11] = ("train", [AttackSpec("Delay", "temperature", start_time=11.0, duration=4.0, params={"delay_sec": 3.0})])
    scenarios[12] = ("train", [AttackSpec("IdentitySpoofing", "rfid", start_time=10.0, duration=12.0)])
    scenarios[13] = ("train", [])  # Benign

    # --- Validation Sessions (14 - 18: 5 sessions) ---
    scenarios[14] = ("val", [])  # Benign
    scenarios[15] = ("val", [
        AttackSpec("DoS", "ultrasonic", start_time=6.0, duration=2.5),
        AttackSpec("IdentitySpoofing", "rfid", start_time=12.0, duration=8.0),
    ])
    scenarios[16] = ("val", [
        AttackSpec("Replay", "motion", start_time=8.0, duration=5.0, params={"delay_offset": 3.5}),
        AttackSpec("Delay", "temperature", start_time=15.0, duration=3.0, params={"delay_sec": 3.2}),
    ])
    scenarios[17] = ("val", [
        AttackSpec("MITM", "temperature", start_time=7.0, duration=4.0),
        AttackSpec("DataFalsification", "motion", start_time=14.0, duration=3.0),
    ])
    scenarios[18] = ("val", [
        AttackSpec("Delay", "ultrasonic", start_time=9.0, duration=3.5, params={"delay_sec": 3.5}),
        AttackSpec("IdentitySpoofing", "rfid", start_time=15.0, duration=10.0),
    ])

    # --- Testing Sessions (19 - 23: 5 sessions) ---
    scenarios[19] = ("test", [])  # Benign
    scenarios[20] = ("test", [
        AttackSpec("DoS", "motion", start_time=7.0, duration=2.5),
        AttackSpec("DataFalsification", "temperature", start_time=14.0, duration=4.0),
    ])
    scenarios[21] = ("test", [
        AttackSpec("Replay", "temperature", start_time=9.0, duration=5.0, params={"delay_offset": 4.0}),
        AttackSpec("IdentitySpoofing", "rfid", start_time=16.0, duration=8.0),
    ])
    scenarios[22] = ("test", [
        AttackSpec("MITM", "ultrasonic", start_time=8.0, duration=4.0),
        AttackSpec("Delay", "ultrasonic", start_time=15.0, duration=3.0, params={"delay_sec": 3.5}),
    ])
    scenarios[23] = ("test", [
        AttackSpec("DataFalsification", "motion", start_time=10.0, duration=3.5),
        AttackSpec("DoS", "ultrasonic", start_time=18.0, duration=2.5),
    ])

    return scenarios


def generate_unified_dataset(
    n_sessions: int = 24,
    session_duration: float = 30.0,
    seed: int = 42,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs all simulation scenarios, extracts features, and builds the dataset."""
    scenarios = define_session_scenarios()
    injector = AttackInjector(seed=seed)

    rows: List[Dict[str, Any]] = []

    for sess_id in range(min(n_sessions, len(scenarios))):
        split_name, attack_specs = scenarios[sess_id]
        sess_seed = seed + sess_id * 100

        # Simulate benign multi-sensor events for this session
        sim = UnifiedSimulator(duration=session_duration, seed=sess_seed)
        raw_events = sim.run()

        # Inject attacks if configured for this session
        if attack_specs:
            events = injector.inject(raw_events, attack_specs)
        else:
            events = raw_events

        # Track per-sensor sequence counters for seq_delta calculation
        last_seq: Dict[str, int] = {}

        for ev in events:
            feat_dict = extract_features_from_event(ev, last_seq)
            row = {
                # Metadata partition
                "session_id": sess_id,
                "split": split_name,
                "timestamp": round(ev.timestamp, 4),
                "sensor_id": ev.sensor_id,
                "topic": ev.topic,
                "event_type": ev.event_type,
                "sequence_number": ev.sequence_number,
                "observed_state": ev.observed_state,
                "output_symbol": ev.output_symbol,
                # Predictive feature partition (X)
                **feat_dict,
                # Quarantined ground-truth partition (y)
                "is_attack": ev.is_attack,
                "attack_type": ev.attack_type,
                "attack_target": ev.attack_target,
                "fault_location": ev.fault_location,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Export complete CSV
    csv_path = out_dir / "four_sensors_dataset.csv"
    df.to_csv(csv_path, index=False)

    # Build manifest
    manifest = {
        "dataset_name": "four_sensors_unified_simulation",
        "total_rows": len(df),
        "total_sessions": n_sessions,
        "duration_per_session_sec": session_duration,
        "seed": seed,
        "splits": {
            s: {
                "rows": int((df["split"] == s).sum()),
                "sessions": sorted([int(x) for x in df[df["split"] == s]["session_id"].unique()]),
                "class_counts": {str(k): int(v) for k, v in df[df["split"] == s]["attack_type"].value_counts().items()},
                "sensor_counts": {str(k): int(v) for k, v in df[df["split"] == s]["sensor_id"].value_counts().items()},
            }
            for s in ("train", "val", "test")
        },
        "overall_class_distribution": {str(k): int(v) for k, v in df["attack_type"].value_counts().items()},
        "overall_sensor_distribution": {str(k): int(v) for k, v in df["sensor_id"].value_counts().items()},
        "observable_feature_columns": list(OBSERVABLE_FEATURE_COLUMNS),
        "ground_truth_columns": list(GROUND_TRUTH_COLUMNS),
        "metadata_columns": list(METADATA_COLUMNS),
    }

    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return df, manifest


def main() -> None:
    print(f"Generating Four-Sensor Unified IoT Dataset...")
    df, manifest = generate_unified_dataset()
    print(f"Generated {len(df)} total rows across {manifest['total_sessions']} sessions.")
    print(f"Saved to: {DEFAULT_OUT_DIR}")
    for split_name, s_info in manifest["splits"].items():
        print(f"  [{split_name.upper()}] Rows: {s_info['rows']} | Sessions: {s_info['sessions']}")
        print(f"    Classes: {s_info['class_counts']}")


if __name__ == "__main__":
    main()

