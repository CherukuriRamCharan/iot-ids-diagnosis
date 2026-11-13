"""Stage 5.3: Two-Tier Hybrid Deep Learning and Formal TFSM Diagnosis Engine.

Combines statistical temporal sequence modeling (Tier 1: LSTM/BiLSTM) with deterministic
mathematical Timed Finite State Machine conformance monitoring (Tier 2: UnifiedDiagnosisEngine).

Architecture Principles:
1. Clear Separation of Concerns:
   - Deep Learning (Tier 1) provides empirical temporal pattern recognition, attack family classification,
     and model confidence scores across sliding 16-event sequence windows.
   - Formal TFSM (Tier 2) provides deterministic conformance verification, timing bound validation,
     suspected transition/state tracking, and mutant candidate fault localization.
2. Zero Cross-Contamination:
   - Neither tier overwrites or fabricates evidence for the other.
   - Formal evidence (expected/observed symbols, deviations, transitions) is never synthesized from DL predictions.
   - DL predictions are never converted directly into formal TFSM conformance proofs.
3. Strict Anti-Leakage:
   - Evaluates strictly the 14 approved observable physical/network features.
   - Strips and ignores all simulator metadata (tampered_by, is_attack, attack_type, fault_location).
4. Explicit Adjudication Outcomes:
   - CONSENSUS: Both tiers detect an attack / anomaly.
   - FORMAL_VIOLATION_DL_NORMAL: Formal FSM detects hard protocol/timing violation; DL predicts Normal.
   - STATISTICAL_ANOMALY: Formal FSM conforms nominally; DL predicts an attack.
   - NOMINAL: Both tiers verify legitimate benign operation.
   - INSUFFICIENT_SEQUENCE: Fewer than 16 events available for DL windowing.
   - MODEL_UNAVAILABLE: Checkpoint loading or inference fails; formal diagnosis runs independently.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from diagnosis_loc.unified_diagnosis import (
    EventDiagnosis,
    SensorDiagnosisReport,
    SystemDiagnosisReport,
    UnifiedDiagnosisEngine,
)
from ids.datasets_four_sensors import (
    FOUR_SENSORS_CLASSES,
    FOUR_SENSORS_PRESERVE_BINARY_COLS,
    SELECTED_OBSERVABLE_FEATURES,
    assert_no_leakage_four_sensors,
)
from ids.models_sequence import build_model

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = ROOT / "results" / "stage_5_hybrid" / "checkpoints"
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "bilstm_multiclass_seed42.pt"
DEFAULT_METADATA_PATH = DEFAULT_CHECKPOINT_DIR / "bilstm_multiclass_seed42.json"


# ----------------------------------------------------------------------
# Structured Output: Unified Hybrid Diagnosis Report
# ----------------------------------------------------------------------
@dataclass
class HybridDiagnosisReport:
    """Comprehensive two-tier diagnostic assessment combining statistical and formal evidence."""

    # 1. Overall Hybrid Status (Required)
    hybrid_outcome: str                 # 'CONSENSUS' | 'FORMAL_VIOLATION_DL_NORMAL' | 'STATISTICAL_ANOMALY' | 'NOMINAL' | 'INSUFFICIENT_SEQUENCE' | 'MODEL_UNAVAILABLE'
    outcome_explanation: str
    agreement: bool                     # True if both tiers concur, False if disagreement
    total_events: int

    # 2. Statistical Deep Learning Tier (Required fields)
    dl_attack_detected: bool
    dl_predicted_category: str          # 'Normal' | 'DoS' | 'Replay' | 'MITM' | 'DataFalsification' | 'Delay' | 'IdentitySpoofing' | 'Unknown' | 'Unavailable'
    dl_confidence_score: float          # Raw softmax score [0.0, 1.0] (Model confidence score, NOT calibrated probability)
    dl_window_count: int

    # 3. Formal TFSM Tier (Required fields)
    formal_anomaly_detected: bool
    formal_conformance: bool
    formal_primary_attack: str
    affected_sensors: List[str]

    # 4. Optional / Defaulted Deep Learning Tier Fields
    dl_window_predictions: List[Dict[str, Any]] = field(default_factory=list)

    # 5. Optional / Defaulted Formal TFSM Tier Fields
    suspected_sensor: Optional[str] = None
    suspected_transition: Optional[str] = None
    suspected_state: Optional[str] = None
    expected_output: Optional[str] = None
    observed_output: Optional[str] = None
    deviation_type: Optional[str] = None
    candidate_fault_mutants: List[str] = field(default_factory=list)
    formal_evidence: List[str] = field(default_factory=list)

    # 6. Optional Underlying Formal Report
    system_diagnosis_report: Optional[SystemDiagnosisReport] = None


# ----------------------------------------------------------------------
# Observable Feature Extraction for Live Inference
# ----------------------------------------------------------------------
def extract_observable_feature_vector(
    event: Dict[str, Any],
    last_seq_by_sensor: Dict[str, int],
    last_global_timestamp: Optional[float] = None,
) -> List[float]:
    """Extracts strictly the 14 approved observable physical/network features.

    Guarantees:
    - Zero simulator watermarks or hidden metadata ('tampered_by', 'is_event_flood').
    - Zero quarantined TFSM states or symbols ('observed_state', 'output_symbol').
    - Zero ground-truth labels ('is_attack', 'attack_type', 'fault_location').
    """
    sid = str(event.get("sensor_id", ""))
    ev_type = str(event.get("event_type", ""))
    seq = int(event.get("sequence_number", 0))
    ts = float(event.get("timestamp", 0.0))
    payload = event.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}

    # 1. inter_arrival
    if "inter_arrival" in event and event["inter_arrival"] is not None:
        ia = float(event["inter_arrival"])
    elif last_global_timestamp is not None and last_global_timestamp > 0.0:
        ia = max(round(ts - last_global_timestamp, 4), 0.0)
    else:
        ia = 0.0

    # 2. payload_size
    psize = float(event.get("payload_size", 48.0))

    # 3. qos
    qos = float(event.get("qos", 0.0))

    # 4. seq_delta
    prev_seq = last_seq_by_sensor.get(sid, seq - 1)
    seq_delta = float(seq - prev_seq)
    last_seq_by_sensor[sid] = seq

    # 5-8. One-hot sensor indicators
    is_motion = 1.0 if sid == "motion" else 0.0
    is_temp = 1.0 if sid == "temperature" else 0.0
    is_ultra = 1.0 if sid == "ultrasonic" else 0.0
    is_rfid = 1.0 if sid == "rfid" else 0.0

    # 9-12. One-hot protocol event indicators
    is_pub = 1.0 if ev_type == "PUBLISH" else 0.0
    is_alert = 1.0 if ev_type == "ALERT" else 0.0
    is_hb = 1.0 if ev_type == "HEARTBEAT" else 0.0
    is_conn = 1.0 if ev_type == "CONNECT" else 0.0

    # 13. numeric_reading (sentinel -1.0 if non-telemetry)
    if "temperature" in payload:
        num_read = float(payload["temperature"])
    elif "distance_cm" in payload:
        num_read = float(payload["distance_cm"])
    elif "motion" in payload:
        num_read = float(payload["motion"])
    elif "reader_gate" in payload:
        num_read = 1.0 if payload.get("status") == "authorized" else 0.0
    else:
        num_read = -1.0

    # 14. has_alert_flag
    has_alert = 1.0 if bool(payload.get("alert") or payload.get("obstacle") or payload.get("motion")) else 0.0

    vec = [
        ia,
        psize,
        qos,
        seq_delta,
        is_motion,
        is_temp,
        is_ultra,
        is_rfid,
        is_pub,
        is_alert,
        is_hb,
        is_conn,
        num_read,
        has_alert,
    ]
    return vec


# ----------------------------------------------------------------------
# Central Hybrid Diagnosis Engine
# ----------------------------------------------------------------------
class HybridDiagnosisEngine:
    """Integrates deep learning sequence classification with formal TFSM diagnosis."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[Path, str]] = None,
        metadata_path: Optional[Union[Path, str]] = None,
        *,
        model: Optional[nn.Module] = None,
        scaler_params: Optional[Dict[str, Any]] = None,
        device_name: str = "cpu",
    ) -> None:
        """Initializes the two-tier hybrid engine.

        Selection Rationale:
        BiLSTM (Seed 42) is selected as production architecture because it achieved highest
        validation Macro-F1 (0.9604 vs 0.9412 for LSTM) on the independent validation split.
        """
        self.device = torch.device(device_name)
        self.fsm_engine = UnifiedDiagnosisEngine()
        self.window_size = 16
        self.feature_names = SELECTED_OBSERVABLE_FEATURES
        self.class_names = FOUR_SENSORS_CLASSES
        self.model: Optional[nn.Module] = model
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_scale: Optional[np.ndarray] = None
        self.model_available = False
        self.model_meta: Dict[str, Any] = {}

        # If a pre-instantiated model was passed (e.g. in unit tests)
        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
            self.model_available = True
            if scaler_params is not None:
                self._load_scaler_params(scaler_params)
            else:
                self.scaler_mean = np.zeros(len(self.feature_names), dtype=np.float32)
                self.scaler_scale = np.ones(len(self.feature_names), dtype=np.float32)
            return

        # Attempt to load checkpoint from path
        ckpt_p = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
        meta_p = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH

        self._load_model_and_metadata(ckpt_p, meta_p)

    def _load_scaler_params(self, params: Dict[str, Any]) -> None:
        """Loads and validates training scaler mean and scale vectors."""
        mean = params.get("mean", [])
        scale = params.get("scale", [])
        if len(mean) == len(self.feature_names) and len(scale) == len(self.feature_names):
            self.scaler_mean = np.array(mean, dtype=np.float32)
            self.scaler_scale = np.array(scale, dtype=np.float32)
            # Avoid division by zero on constant columns
            self.scaler_scale[self.scaler_scale == 0.0] = 1.0
        else:
            self.scaler_mean = np.zeros(len(self.feature_names), dtype=np.float32)
            self.scaler_scale = np.ones(len(self.feature_names), dtype=np.float32)

    def _load_model_and_metadata(self, pt_path: Path, json_path: Path) -> None:
        """Loads PyTorch model weights and verifies metadata compatibility."""
        if not pt_path.is_file():
            self.model_available = False
            return

        try:
            ckpt = torch.load(pt_path, map_location=self.device)
            model_name = ckpt.get("model_name", "BiLSTM")
            config = ckpt.get("config", {})

            # Instantiate model architecture
            loaded_model = build_model(
                model_name,
                num_features=config.get("num_features", len(self.feature_names)),
                hidden_size=config.get("hidden_size", 64),
                num_layers=config.get("num_layers", 2),
                num_classes=config.get("num_classes", len(self.class_names)),
                dropout=config.get("dropout", 0.2),
            )
            loaded_model.load_state_dict(ckpt["state_dict"])
            loaded_model.to(self.device)
            loaded_model.eval()

            self.model = loaded_model
            self.model_available = True
            self.model_meta = ckpt

            # Load metadata JSON if available
            if json_path.is_file():
                with open(json_path, encoding="utf-8") as f:
                    meta_json = json.load(f)
                s_params = meta_json.get("meta", {}).get("scaler_params", {})
                self._load_scaler_params(s_params)
            else:
                self.scaler_mean = np.zeros(len(self.feature_names), dtype=np.float32)
                self.scaler_scale = np.ones(len(self.feature_names), dtype=np.float32)

        except Exception as exc:
            self.model_available = False
            self.model = None

    def transform_features(self, X_raw: np.ndarray) -> np.ndarray:
        """Applies training-only standardization matching MaskedStandardScaler."""
        if self.scaler_mean is None or self.scaler_scale is None:
            return X_raw.astype(np.float32)

        X_norm = X_raw.copy().astype(np.float32)
        for i, col in enumerate(self.feature_names):
            if col in FOUR_SENSORS_PRESERVE_BINARY_COLS:
                # Binary availability/event indicators are preserved without scaling
                continue
            elif col == "numeric_reading":
                # Masked sentinel (-1.0) mapped to 0.0 after normalization
                mask = X_norm[:, i] != -1.0
                X_norm[mask, i] = (X_norm[mask, i] - self.scaler_mean[i]) / self.scaler_scale[i]
                X_norm[~mask, i] = 0.0
            else:
                X_norm[:, i] = (X_norm[:, i] - self.scaler_mean[i]) / self.scaler_scale[i]

        return X_norm

    def diagnose(
        self,
        events: Union[List[Dict[str, Any]], List[Any]],
    ) -> HybridDiagnosisReport:
        """Executes independent Tier-1 (DL) and Tier-2 (FSM) evaluation and reconciles adjudication."""
        total_events = len(events)

        # --------------------------------------------------------------
        # 1. Independent Formal TFSM Diagnosis (Tier 2)
        # --------------------------------------------------------------
        fsm_report: SystemDiagnosisReport = self.fsm_engine.diagnose_events(events)

        formal_anomaly = bool(fsm_report.overall_anomaly_detected)
        formal_primary = str(fsm_report.primary_predicted_attack)
        affected_sensors = list(fsm_report.affected_sensors)

        # Extract formal evidence fields from sensor reports
        suspected_sensor = affected_sensors[0] if affected_sensors else None
        suspected_st = None
        suspected_tr = None
        exp_out = None
        obs_out = None
        dev_type = None
        all_faults: List[str] = []
        all_evidence: List[str] = []
        conformance = True

        for sid, srep in fsm_report.sensor_reports.items():
            if not srep.fsm_conformance:
                conformance = False
            if srep.anomaly_detected:
                all_faults.extend(srep.candidate_faults)
                all_evidence.extend(srep.evidence)
                if suspected_st is None and srep.suspected_state:
                    suspected_st = srep.suspected_state
                if suspected_tr is None and srep.suspected_transition:
                    suspected_tr = srep.suspected_transition
                if exp_out is None and srep.expected_output:
                    exp_out = srep.expected_output
                if obs_out is None and srep.observed_output:
                    obs_out = srep.observed_output

        # Extract first non-NONE deviation type
        for ed in fsm_report.event_diagnoses:
            if ed.deviation_type != "NONE":
                dev_type = ed.deviation_type
                break

        candidate_faults = sorted(list(set(all_faults)))

        # --------------------------------------------------------------
        # 2. Clean Observable Events for Deep Learning (Tier 1)
        # --------------------------------------------------------------
        clean_events: List[Dict[str, Any]] = []
        for e in events:
            if hasattr(e, "to_observable_dict"):
                clean_events.append(e.to_observable_dict())
            elif isinstance(e, dict):
                clean_e = copy.deepcopy(e)
                # Strip all ground truth and simulator metadata
                for qk in (
                    "is_attack",
                    "attack_type",
                    "attack_target",
                    "fault_location",
                    "original_payload",
                    "original_timestamp",
                    "tampered_by",
                    "is_event_flood",
                    "has_tamper_flag",
                ):
                    clean_e.pop(qk, None)
                clean_events.append(clean_e)

        # Sort strictly chronologically by timestamp and sequence number
        clean_events.sort(key=lambda x: (float(x.get("timestamp", 0.0)), int(x.get("sequence_number", 0))))

        # --------------------------------------------------------------
        # 3. Check Sequence Length & Model Availability Boundaries
        # --------------------------------------------------------------
        if not self.model_available or self.model is None:
            explanation = (
                "Deep learning sequence model is unavailable. "
                "Formal TFSM conformance and fault localization executed independently."
            )
            return HybridDiagnosisReport(
                hybrid_outcome="MODEL_UNAVAILABLE",
                outcome_explanation=explanation,
                agreement=False,
                total_events=total_events,
                dl_attack_detected=False,
                dl_predicted_category="Unavailable",
                dl_confidence_score=0.0,
                dl_window_count=0,
                formal_anomaly_detected=formal_anomaly,
                formal_conformance=conformance,
                formal_primary_attack=formal_primary,
                affected_sensors=affected_sensors,
                suspected_sensor=suspected_sensor,
                suspected_transition=suspected_tr,
                suspected_state=suspected_st,
                expected_output=exp_out,
                observed_output=obs_out,
                deviation_type=dev_type,
                candidate_fault_mutants=candidate_faults,
                formal_evidence=all_evidence[:8],
                system_diagnosis_report=fsm_report,
            )

        if total_events < self.window_size:
            explanation = (
                f"Insufficient sequence length: received {total_events} events, but sliding window "
                f"requires at least {self.window_size} events for recurrent inference. "
                f"Formal TFSM conformance executed independently."
            )
            return HybridDiagnosisReport(
                hybrid_outcome="INSUFFICIENT_SEQUENCE",
                outcome_explanation=explanation,
                agreement=False,
                total_events=total_events,
                dl_attack_detected=False,
                dl_predicted_category="Unknown",
                dl_confidence_score=0.0,
                dl_window_count=0,
                formal_anomaly_detected=formal_anomaly,
                formal_conformance=conformance,
                formal_primary_attack=formal_primary,
                affected_sensors=affected_sensors,
                suspected_sensor=suspected_sensor,
                suspected_transition=suspected_tr,
                suspected_state=suspected_st,
                expected_output=exp_out,
                observed_output=obs_out,
                deviation_type=dev_type,
                candidate_fault_mutants=candidate_faults,
                formal_evidence=all_evidence[:8],
                system_diagnosis_report=fsm_report,
            )

        # --------------------------------------------------------------
        # 4. Deep Learning Sequence Windowing & Inference (Tier 1)
        # --------------------------------------------------------------
        feature_rows: List[List[float]] = []
        last_seq_by_sensor: Dict[str, int] = {}
        last_t = 0.0

        for ev in clean_events:
            vec = extract_observable_feature_vector(ev, last_seq_by_sensor, last_t)
            feature_rows.append(vec)
            last_t = float(ev.get("timestamp", 0.0))

        X_raw = np.array(feature_rows, dtype=np.float32)
        X_norm = self.transform_features(X_raw)

        # Build sliding windows of size 16 (stride 1)
        windows: List[np.ndarray] = []
        n_windows = len(X_norm) - self.window_size + 1
        for start_idx in range(n_windows):
            windows.append(X_norm[start_idx : start_idx + self.window_size])

        X_tensor = torch.from_numpy(np.stack(windows, axis=0)).float().to(self.device)

        with torch.no_grad():
            logits = self.model(X_tensor)
            probas = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probas, axis=-1)

        window_predictions: List[Dict[str, Any]] = []
        for w_idx in range(n_windows):
            p_cls = self.class_names[preds[w_idx]]
            conf = float(probas[w_idx, preds[w_idx]])
            window_predictions.append(
                {
                    "window_index": w_idx,
                    "end_event_index": w_idx + self.window_size - 1,
                    "predicted_class": p_cls,
                    "confidence_score": round(conf, 4),
                }
            )

        # Identify primary DL prediction: prioritize highest confidence attack window, or Normal
        attack_windows = [w for w in window_predictions if w["predicted_class"] != "Normal"]
        if attack_windows:
            best_attack = max(attack_windows, key=lambda w: w["confidence_score"])
            dl_category = best_attack["predicted_class"]
            dl_confidence = best_attack["confidence_score"]
            dl_attack_detected = True
        else:
            normal_windows = [w for w in window_predictions if w["predicted_class"] == "Normal"]
            best_normal = max(normal_windows, key=lambda w: w["confidence_score"]) if normal_windows else window_predictions[-1]
            dl_category = "Normal"
            dl_confidence = best_normal["confidence_score"]
            dl_attack_detected = False

        # --------------------------------------------------------------
        # 5. Hybrid Adjudication Logic (Reconciling Tiers)
        # --------------------------------------------------------------
        if formal_anomaly and dl_attack_detected:
            # Case A: CONSENSUS
            hybrid_outcome = "CONSENSUS"
            agreement = True
            explanation = (
                f"CONSENSUS: Both diagnostic tiers concur. Formal TFSM detected an anomaly "
                f"({formal_primary}, deviation={dev_type}) and deep learning sequence model "
                f"classified an attack as '{dl_category}' with model confidence score {dl_confidence:.4f}."
            )

        elif formal_anomaly and not dl_attack_detected:
            # Case B: FORMAL_VIOLATION_DL_NORMAL
            hybrid_outcome = "FORMAL_VIOLATION_DL_NORMAL"
            agreement = False
            explanation = (
                f"FORMAL VIOLATION (DL DISAGREEMENT): Formal TFSM detected a hard protocol, timing, "
                f"or sequence deviation ({dev_type}, primary={formal_primary}) on sensor '{suspected_sensor}', "
                f"but deep learning sequence model predicted Normal with confidence {dl_confidence:.4f}. "
                f"Formal mathematical violation proof is strictly preserved."
            )

        elif not formal_anomaly and dl_attack_detected:
            # Case C: STATISTICAL_ANOMALY
            hybrid_outcome = "STATISTICAL_ANOMALY"
            agreement = False
            explanation = (
                f"STATISTICAL ANOMALY: Deep learning sequence model detected temporal patterns "
                f"resembling '{dl_category}' with model confidence score {dl_confidence:.4f}, "
                f"but all observable events nominally conform to formal TFSM transitions and timing bounds. "
                f"Flagged for operator inspection; no formal proof of an attack."
            )

        else:
            # Case D: NOMINAL
            hybrid_outcome = "NOMINAL"
            agreement = True
            explanation = (
                f"NOMINAL: Both tiers verify legitimate benign operation. "
                f"Formal TFSM conformance passed across all sensors, and deep learning "
                f"sequence model classified traffic as Normal with confidence {dl_confidence:.4f}."
            )

        return HybridDiagnosisReport(
            hybrid_outcome=hybrid_outcome,
            outcome_explanation=explanation,
            agreement=agreement,
            total_events=total_events,
            dl_attack_detected=dl_attack_detected,
            dl_predicted_category=dl_category,
            dl_confidence_score=dl_confidence,
            dl_window_count=n_windows,
            dl_window_predictions=window_predictions[:10],
            formal_anomaly_detected=formal_anomaly,
            formal_conformance=conformance,
            formal_primary_attack=formal_primary,
            affected_sensors=affected_sensors,
            suspected_sensor=suspected_sensor,
            suspected_transition=suspected_tr,
            suspected_state=suspected_st,
            expected_output=exp_out,
            observed_output=obs_out,
            deviation_type=dev_type,
            candidate_fault_mutants=candidate_faults,
            formal_evidence=all_evidence[:8],
            system_diagnosis_report=fsm_report,
        )

