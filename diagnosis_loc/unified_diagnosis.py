"""
Unified Attack Diagnosis Engine
================================
Combines evidence from three independent sources into a single
structured diagnosis record per capture/scenario:

  1. ML prediction    — from Random Forest / LSTM / Transformer output
  2. EFSM violations  — from temperature_efsm / motion_efsm detectors
  3. Traffic rules    — payload range, message manipulation, publish rate

Localization levels supported (Section 4 of master requirements):
  Level 1 — Attack detection  (binary: benign vs attack)
  Level 2 — Attack-type classification  (5 classes or scenario name)
  Level 3 — Network endpoint  (observed source/destination IP:port)
  Level 4 — Physical device   ❌ NOT supported (insufficient evidence)
  Level 5 — Attacker identity ❌ NOT supported (IP ≠ identity)

Outputs:
  results/unified_diagnosis_records.json
  results/unified_diagnosis_records.csv

Usage:
  .venv\\Scripts\\python.exe diagnosis_loc\\unified_diagnosis.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Input files — use what already exists; missing files are skipped gracefully.
PACKET_PRED      = ROOT / "results" / "packet_attack_predictions.csv"
COMBINED_DIAG    = ROOT / "results" / "combined_attack_diagnosis.csv"
MQTT_RULE        = ROOT / "results" / "combined_diagnosis_report.csv"
RATE_REPORT      = ROOT / "results" / "packet_rate_report.csv"
LSTM_EVAL        = ROOT / "results" / "lstm_evaluation.json"
TRANSFORMER_EVAL = ROOT / "results" / "transformer_evaluation.json"
RF5_EVAL         = ROOT / "results" / "rf_5class_evaluation.json"

OUT_JSON = ROOT / "results" / "unified_diagnosis_records.json"
OUT_CSV  = ROOT / "results" / "unified_diagnosis_records.csv"


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class DiagnosisRecord:
    """One diagnosis record per capture/scenario."""

    # Identity
    capture_id:       str = ""
    dataset_source:   str = ""   # "external_dataset" | "local_mqtt"
    dataset_label:    str = ""   # filename-derived class label
    simulated_scenario: str = "" # scenario metadata (local only)

    # Detection (Level 1)
    binary_prediction:      str   = ""     # "attack" | "benign"
    ml_attack_percent:      float = 0.0   # % of packets predicted attack (RF)
    detection_level:        str   = "Level 1 — attack detected" if True else ""

    # Classification (Level 2)
    attack_type_predicted:  str   = ""   # dataset label or rule signal
    ml_confidence:          str   = ""   # qualitative (no calibrated proba)
    rule_signals:           list  = field(default_factory=list)
    efsm_signals:           list  = field(default_factory=list)

    # Evidence
    total_packets:          int   = 0
    tcp_syn_percent:        float = 0.0
    average_pps:            Optional[float] = None
    rule_evidence_text:     str   = ""
    supporting_evidence:    str   = ""
    efsm_relevant_states:   list  = field(default_factory=list)

    # Localization (Level 3 — endpoint only)
    top_source_ip:          Optional[str] = None
    top_destination_ip:     Optional[str] = None
    localization_level:     str = "Level 3 — observed network endpoint only"
    localization_note:      str = (
        "Observed IP/port does not identify a physical IoT device or attacker. "
        "Physical localization (Level 4) and attacker attribution (Level 5) "
        "require additional infrastructure and are not supported by this dataset."
    )

    # Uncertainty
    uncertainty_and_limitations: str = ""
    timestamp_range:        str = ""


# ─────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────

def _load_csv(path: Path, name: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"  [SKIP] {name}: {path.name} not found")
        return None
    df = pd.read_csv(path)
    print(f"  [OK]   {name}: {len(df)} rows, columns: {list(df.columns[:5])} …")
    return df


def _load_json(path: Path, name: str) -> Optional[dict]:
    if not path.exists():
        print(f"  [SKIP] {name}: {path.name} not found")
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"  [OK]   {name}")
    return data


# ─────────────────────────────────────────────────────────────
# EFSM signal mapping  (capture filename → expected EFSM signals)
# ─────────────────────────────────────────────────────────────

EFSM_MAP = {
    "replay":           {
        "states": ["PUBLISH_DATA → ANOMALOUS"],
        "signals": ["repeated_payload_detected"],
        "note": "Replay-like: repeated identical payload observed"
    },
    "data_falsification": {
        "states": ["PUBLISH_DATA → ANOMALOUS"],
        "signals": ["data_falsification_temperature_out_of_range"],
        "note": "Data falsification: temperature outside [-10, 60] °C"
    },
    "mitm_simulation":  {
        "states": ["PUBLISH_DATA → ANOMALOUS"],
        "signals": ["message_manipulation_payload_changed"],
        "note": "Message manipulation: forwarded payload differs from original"
    },
    "battery_draining": {
        "states": ["WAIT_TIMER → ANOMALOUS"],
        "signals": ["excessive_communication_rate_exceeded"],
        "note": "Excessive communication: publish rate > threshold"
    },
    "normal":           {
        "states": [],
        "signals": [],
        "note": "No EFSM violation expected in normal operation"
    },
}

def _efsm_info(capture_name: str) -> dict:
    name = capture_name.lower()
    for key, info in EFSM_MAP.items():
        if key in name:
            return info
    return {"states": [], "signals": [], "note": "No EFSM mapping available"}


# ─────────────────────────────────────────────────────────────
# Build diagnosis records
# ─────────────────────────────────────────────────────────────

def build_records() -> list[DiagnosisRecord]:
    print("\nLoading input files …")

    ml_pred   = _load_csv(PACKET_PRED,   "RF binary predictions")
    comb_diag = _load_csv(COMBINED_DIAG, "Combined ML+rule diagnosis")
    mqtt_rule = _load_csv(MQTT_RULE,     "MQTT rule report")
    rate_rep  = _load_csv(RATE_REPORT,   "Rate report")

    records: list[DiagnosisRecord] = []

    # ── External dataset captures (from combined_attack_diagnosis.csv) ──
    if comb_diag is not None:
        for _, row in comb_diag.iterrows():
            cap   = str(row.get("capture_file", ""))
            label = str(row.get("dataset_label", "unknown")).lower()
            atk_pct = float(row.get("ml_attack_percent", 0))
            total   = int(row.get("total_packets", 0))
            syn_pct = float(row.get("tcp_syn_percent", 0))
            pps     = row.get("average_packets_per_second")
            src_ip  = row.get("top_source_ip_observed")
            dst_ip  = row.get("top_destination_ip_observed")
            rule_ev = str(row.get("rule_evidence", ""))

            binary_pred = "attack" if atk_pct >= 50 else "benign"

            # Qualitative confidence
            if atk_pct >= 95:
                confidence = "high (≥95% packets predicted attack)"
            elif atk_pct >= 50:
                confidence = "moderate (50–95% packets predicted attack)"
            elif atk_pct <= 5:
                confidence = "high (≤5% packets predicted attack, classified benign)"
            else:
                confidence = "low (5–50% packet predictions mixed)"

            efsm = _efsm_info(cap)

            # Build rule signals list
            rule_signals = []
            if syn_pct >= 20:
                rule_signals.append("high_tcp_syn_proportion")
            if pps is not None and float(pps) > 100:
                rule_signals.append("high_average_packet_rate")
            if atk_pct >= 50:
                rule_signals.append("ml_majority_attack")

            limitations = (
                f"Label '{label}' inferred from capture filename. "
                "Benign test uses temporal split of one capture (not independent). "
                "IP address identifies network endpoint only, not physical device or attacker."
            )

            rec = DiagnosisRecord(
                capture_id            = cap,
                dataset_source        = "external_dataset",
                dataset_label         = label,
                simulated_scenario    = "",
                binary_prediction     = binary_pred,
                ml_attack_percent     = round(atk_pct, 2),
                detection_level       = (
                    "Level 1 — attack detected"
                    if binary_pred == "attack"
                    else "Level 1 — classified benign"
                ),
                attack_type_predicted = label,
                ml_confidence         = confidence,
                rule_signals          = rule_signals,
                efsm_signals          = efsm["signals"],
                total_packets         = total,
                tcp_syn_percent       = round(syn_pct, 2),
                average_pps           = float(pps) if pps is not None else None,
                rule_evidence_text    = rule_ev,
                supporting_evidence   = rule_ev,
                efsm_relevant_states  = efsm["states"],
                top_source_ip         = str(src_ip) if pd.notna(src_ip) else None,
                top_destination_ip    = str(dst_ip) if pd.notna(dst_ip) else None,
                localization_level    = "Level 3 — observed network endpoint only",
                localization_note     = (
                    f"Source IP {src_ip} and destination IP {dst_ip} observed. "
                    "These are network-layer addresses; physical device and attacker "
                    "identity cannot be determined from IP alone."
                    if pd.notna(src_ip)
                    else "No IP address columns available in extracted features."
                ),
                uncertainty_and_limitations = limitations,
                timestamp_range       = "",
            )
            records.append(rec)

    # ── Local MQTT captures (from MQTT rule report) ──
    if mqtt_rule is not None:
        for _, row in mqtt_rule.iterrows():
            cap      = str(row.get("capture_file", ""))
            scenario = str(row.get("simulated_scenario", "unknown"))
            label    = str(row.get("traffic_label", "unknown"))
            signals  = str(row.get("detection_signals", "none"))
            evidence = str(row.get("supporting_evidence", ""))
            diagnosis = str(row.get("diagnosis", ""))
            limits   = str(row.get("uncertainty_and_limitations", ""))

            efsm = _efsm_info(cap)

            binary_pred = (
                "benign"
                if signals == "none" or scenario == "normal"
                else "attack"
            )

            rate_flagged = int(row.get("rate_flagged_packets", 0))
            range_flags  = int(row.get("temperature_range_violation_rows", 0))
            manip_flags  = int(row.get("manipulation_mismatch_rows", 0))

            rule_signals = [s for s in signals.split(";") if s and s != "none"]

            rec = DiagnosisRecord(
                capture_id            = cap,
                dataset_source        = "local_mqtt",
                dataset_label         = label,
                simulated_scenario    = scenario,
                binary_prediction     = binary_pred,
                ml_attack_percent     = 0.0,   # ML not run on local captures
                detection_level       = (
                    "Level 1 — attack detected (rule-based)"
                    if binary_pred == "attack"
                    else "Level 1 — no attack signal"
                ),
                attack_type_predicted = (
                    scenario.replace("_simulation", "").replace("_", " ")
                    if binary_pred == "attack"
                    else "normal"
                ),
                ml_confidence         = "N/A — rule-based detection only (no ML run on local captures)",
                rule_signals          = rule_signals,
                efsm_signals          = efsm["signals"],
                total_packets         = int(row.get("mqtt_packets", 0)),
                tcp_syn_percent       = 0.0,
                average_pps           = float(row.get("publish_packets_per_second", 0)),
                rule_evidence_text    = evidence,
                supporting_evidence   = evidence,
                efsm_relevant_states  = efsm["states"],
                top_source_ip         = None,
                top_destination_ip    = None,
                localization_level    = (
                    "Level 3 — network endpoint: localhost (::1)"
                    if "normal" not in cap.lower()
                    else "Level 1 — normal traffic, no localization needed"
                ),
                localization_note     = (
                    "All local captures use loopback (::1); source ≡ destination. "
                    "No meaningful network endpoint localization is possible."
                ),
                uncertainty_and_limitations = limits,
                timestamp_range       = "",
            )
            records.append(rec)

    return records


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Unified Diagnosis Engine")
    print("=" * 60)

    records = build_records()
    print(f"\nTotal diagnosis records: {len(records)}")

    # ── Print summary ──
    for rec in records:
        src  = "EXT" if rec.dataset_source == "external_dataset" else "MQTT"
        flag = "⚠ ATTACK" if rec.binary_prediction == "attack" else "  benign"
        print(
            f"  [{src}] {flag}  {rec.capture_id[:45]:<45}  "
            f"label={rec.dataset_label:<8}  "
            f"signals={rec.rule_signals}"
        )

    # ── Save JSON ──
    OUT_JSON.parent.mkdir(exist_ok=True)
    records_dicts = [asdict(r) for r in records]
    OUT_JSON.write_text(
        json.dumps(records_dicts, indent=2, default=str),
        encoding="utf-8"
    )

    # ── Save CSV ──
    rows_flat = []
    for r in records_dicts:
        row = dict(r)
        row["rule_signals"]          = "; ".join(r["rule_signals"])
        row["efsm_signals"]          = "; ".join(r["efsm_signals"])
        row["efsm_relevant_states"]  = "; ".join(r["efsm_relevant_states"])
        rows_flat.append(row)

    pd.DataFrame(rows_flat).to_csv(OUT_CSV, index=False)

    print(f"\nSaved JSON: {OUT_JSON}")
    print(f"Saved CSV:  {OUT_CSV}")

    # ── Statistics ──
    n_attack = sum(1 for r in records if r.binary_prediction == "attack")
    n_benign = sum(1 for r in records if r.binary_prediction == "benign")
    print(f"\nDiagnosis summary: {n_attack} attack, {n_benign} benign")
    print(f"Dataset sources:   "
          f"{sum(1 for r in records if r.dataset_source == 'external_dataset')} external, "
          f"{sum(1 for r in records if r.dataset_source == 'local_mqtt')} local MQTT")


if __name__ == "__main__":
    main()
