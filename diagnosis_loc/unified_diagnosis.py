"""Formal FSM-Based Diagnosis and Fault Localization Engine.

Connects the sensor-specific Timed Finite State Machine (TFSM) models to an online
stream monitor to perform:
1. Formal FSM Conformance Monitoring: verifies transitions, state sequence, and outputs.
2. Traffic & Stream Anomaly Detection: inspects sensor-isolated timing constraints, sequence progression, and physical bounds.
3. Candidate Attack Classification: identifies which scenario (DoS, Replay, MITM, DFA, Delay, Spoofing) explains the symptoms.
4. Fault Localization: pinpoints the affected sensor, violated transition, suspected state, and candidate mutants.

Sensor Isolation & Anti-Leakage Guarantee:
- Each sensor maintains its own independent timing baseline, sequence tracking, and FSM runtime.
- Operates STRICTLY on observable event features. Ground-truth labels (is_attack, attack_type,
  fault_location, original_payload) are NEVER accessed or used during inference.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from models_fsm import motion_sensor, rfid_sensor, temperature_sensor, ultrasonic_sensor
from models_fsm.tfsm import TFSM, Transition


# ----------------------------------------------------------------------
# Physical Domain Bounds (legitimately known specification constraints)
# ----------------------------------------------------------------------
PHYSICAL_BOUNDS = {
    "temperature": {"min": -20.0, "max": 70.0, "unit": "Celsius"},
    "ultrasonic": {"min": 2.0, "max": 450.0, "unit": "cm"},
    "motion": {"min_conf": 0.0, "max_conf": 1.0},
}

# Sensor-specific timing baselines and keepalive timeout thresholds
TIMEOUT_THRESHOLDS = {
    "motion": 2.2,
    "temperature": 2.3,
    "ultrasonic": 2.05,
    "rfid": 2.8,
}

NOMINAL_PERIODS = {
    "motion": 1.0,
    "temperature": 1.5,
    "ultrasonic": 0.8,
    "rfid": 2.5,
}


# ----------------------------------------------------------------------
# Structured Output Data Classes
# ----------------------------------------------------------------------
@dataclass
class EventDiagnosis:
    """Diagnostic assessment for a single observable event."""

    timestamp: float
    sensor_id: str
    event_type: str
    sequence_number: int
    is_anomaly: bool
    fsm_conformance: bool
    deviation_type: str                   # 'NONE' | 'OUTPUT_MISMATCH' | 'TIMING_VIOLATION' | 'SEQUENCE_REGRESSION' | 'REPLAY_DUPLICATE' | 'BURST_FLOOD' | 'PHYSICAL_BOUNDS' | 'TAMPERING' | 'SPOOFING'
    suspected_state: Optional[str] = None
    suspected_transition: Optional[str] = None
    expected_output: Optional[str] = None
    observed_output: Optional[str] = None
    candidate_faults: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    diagnosis_confidence: float = 1.0


@dataclass
class SensorDiagnosisReport:
    """Aggregate formal diagnosis and localization report for a single sensor."""

    sensor_id: str
    anomaly_detected: bool
    fsm_conformance: bool
    total_events: int
    anomalous_events_count: int
    predicted_attack_category: str        # 'Normal' | 'DoS' | 'Replay' | 'MITM' | 'DataFalsification' | 'Delay' | 'IdentitySpoofing'
    suspected_state: Optional[str] = None
    suspected_transition: Optional[str] = None
    expected_output: Optional[str] = None
    observed_output: Optional[str] = None
    candidate_faults: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    diagnosis_confidence: float = 1.0


@dataclass
class SystemDiagnosisReport:
    """Comprehensive diagnosis report across all sensors in an IoT session."""

    overall_anomaly_detected: bool
    affected_sensors: List[str]
    sensor_reports: Dict[str, SensorDiagnosisReport]
    primary_predicted_attack: str
    event_diagnoses: List[EventDiagnosis]


# ----------------------------------------------------------------------
# Sensor Monitor Runtime (Fully Isolated Per Sensor)
# ----------------------------------------------------------------------
class SensorMonitor:
    """Tracks the nominal state of a single sensor and checks conformance."""

    def __init__(self, sensor_id: str, fsm_module: Any) -> None:
        self.sensor_id = sensor_id
        self.mod = fsm_module
        self.machine: TFSM = self.mod.build_original()
        self.state: str = self.machine.init
        self.last_timestamp: float = 0.0
        self.last_seq: Optional[int] = None
        self.seen_seqs: Set[int] = set()
        self.timeout_threshold: float = TIMEOUT_THRESHOLDS.get(sensor_id, 2.2)

    def reconstruct_input_symbol(self, event: Dict[str, Any]) -> str:
        """Infers the legitimate formal FSM input symbol from observable message content."""
        ev_type = event.get("event_type", "")
        payload = event.get("payload", {})

        if ev_type == "CONNECT":
            action = payload.get("action", "")
            if "wifi" in action:
                return "i2"  # WiFi connected signal ws(1)
            return "i4"      # MQTT connected signal cc(1)

        if ev_type == "HEARTBEAT":
            return "i5"      # Health check / alive

        if ev_type == "FLOOD":
            return "i6"      # Unexpected request / malformed message

        # Domain Telemetry Inputs
        if self.sensor_id == "motion":
            is_motion = bool(payload.get("motion", 0))
            return "i11" if is_motion else "i5"

        elif self.sensor_id == "temperature":
            is_alert = bool(payload.get("alert", False)) or float(payload.get("temperature", 20.0)) > 30.0
            return "i12" if is_alert else "i11"

        elif self.sensor_id == "ultrasonic":
            is_obs = bool(payload.get("obstacle", False)) or float(payload.get("distance_cm", 100.0)) < 30.0
            return "i12" if is_obs else "i11"

        elif self.sensor_id == "rfid":
            uid = payload.get("card_uid", "")
            # Only legitimate authorized card produces i11
            is_authorized = (uid == "A1B2C3D4")
            return "i11" if is_authorized else "i12"

        return "i6"

    def evaluate_event(self, event: Dict[str, Any]) -> EventDiagnosis:
        """Evaluates an observable event against the formal FSM and isolated sensor baselines."""
        ts = float(event.get("timestamp", 0.0))
        seq = int(event.get("sequence_number", 0))
        obs_out = event.get("output_symbol", "")
        payload = event.get("payload", {})
        ev_type = event.get("event_type", "")

        # Compute isolated elapsed time for THIS sensor only
        if self.last_timestamp > 0.0:
            sensor_dt = ts - self.last_timestamp
        else:
            raw_ia = float(event.get("inter_arrival", 0.0))
            sensor_dt = raw_ia if raw_ia > 0.0 else 0.0

        is_anomaly = False
        conformance = True
        dev_type = "NONE"
        suspected_st = None
        suspected_tr = None
        exp_out = None
        candidate_faults = []
        evidence = []
        conf = 1.0

        # --------------------------------------------------------------
        # 1. Isolated Sequence & Rate Checks
        # --------------------------------------------------------------
        if seq in self.seen_seqs:
            # Genuine Replay: Duplicate sequence number already seen
            is_anomaly = True
            dev_type = "REPLAY_DUPLICATE"
            candidate_faults.extend(["replayed_stale_sequence", "RA_t9"])
            evidence.append(f"Duplicate sequence number on {self.sensor_id}: seq={seq} was already seen (replay detected)")
            conf = 0.95
        elif self.last_seq is not None and seq <= self.last_seq:
            # Sequence regression (can occur from out-of-order delayed packets or replay)
            is_anomaly = True
            dev_type = "SEQUENCE_REGRESSION"
            candidate_faults.append("replayed_stale_sequence")
            evidence.append(f"Non-monotonic sequence number on {self.sensor_id}: seq={seq} <= last_seq={self.last_seq}")
        elif self.last_seq is not None and (seq - self.last_seq) > 5 and ev_type == "FLOOD":
            is_anomaly = True
            dev_type = "BURST_FLOOD"
            candidate_faults.append("dos_queue_exhaustion")
            evidence.append(f"Sequence surge on FLOOD frame for {self.sensor_id}: delta={seq - self.last_seq}")
            conf = 0.95

        self.seen_seqs.add(seq)

        # Check Burst Flood Rate (Sensor-isolated interval check)
        if ev_type == "FLOOD" or (self.last_timestamp > 0.5 and sensor_dt < 0.08 and ev_type != "CONNECT"):
            is_anomaly = True
            if dev_type == "NONE" or ev_type == "FLOOD":
                dev_type = "BURST_FLOOD"
            candidate_faults.append("dos_queue_exhaustion")
            evidence.append(f"High-frequency burst rate on {self.sensor_id}: dt={sensor_dt:.4f}s")
            conf = max(conf, 0.92)

        # Check Message Delay Exceeding Keepalive Timeout
        if sensor_dt > self.timeout_threshold and ev_type not in ("CONNECT", "FLOOD"):
            is_anomaly = True
            if dev_type == "NONE":
                dev_type = "TIMING_VIOLATION"
            candidate_faults.extend(["delay_timeout_exceeded", "BD_s8", "BD_s9"])
            evidence.append(f"Timing violation on {self.sensor_id}: interval dt={sensor_dt:.4f}s exceeds timeout threshold {self.timeout_threshold}s")
            conf = 0.85

        # --------------------------------------------------------------
        # 2. Domain Physical Bounds & In-Flight Tampering Checks
        # --------------------------------------------------------------
        if self.sensor_id == "temperature" and "temperature" in payload:
            temp_val = float(payload["temperature"])
            if temp_val < PHYSICAL_BOUNDS["temperature"]["min"] or temp_val > PHYSICAL_BOUNDS["temperature"]["max"]:
                is_anomaly = True
                dev_type = "PHYSICAL_BOUNDS"
                candidate_faults.extend(["dfa_sensor_falsification", "DFA_t21", "DFA_t23"])
                evidence.append(f"Temperature reading {temp_val}°C violates physical domain bounds [-20, 70]")
                conf = 0.95
            elif temp_val > 30.0 and obs_out == "o11":
                # In-flight MITM tampering: payload reads high alert temp but output symbol is normal o11
                is_anomaly = True
                dev_type = "TAMPERING"
                candidate_faults.append("mitm_intercept_tamper")
                evidence.append(f"Semantic inconsistency on temperature: reading {temp_val}°C should trigger alert but output is o11")
                conf = 0.90

        elif self.sensor_id == "ultrasonic" and "distance_cm" in payload:
            dist_val = float(payload["distance_cm"])
            if dist_val < PHYSICAL_BOUNDS["ultrasonic"]["min"] or dist_val > PHYSICAL_BOUNDS["ultrasonic"]["max"]:
                is_anomaly = True
                dev_type = "PHYSICAL_BOUNDS"
                candidate_faults.extend(["dfa_sensor_falsification", "DFA_t21", "DFA_t23"])
                evidence.append(f"Ultrasonic distance {dist_val}cm violates physical domain bounds [2, 450]")
                conf = 0.95
            elif (dist_val < 30.0 or payload.get("obstacle") is True) and obs_out == "o11":
                # In-flight MITM tampering: payload reads obstacle but output symbol is normal o11
                is_anomaly = True
                dev_type = "TAMPERING"
                candidate_faults.append("mitm_intercept_tamper")
                evidence.append("Semantic inconsistency on ultrasonic: payload claims obstacle but output symbol is o11")
                conf = 0.90

        elif self.sensor_id == "motion" and "confidence" in payload:
            conf_val = float(payload["confidence"])
            if conf_val < 0.0 or conf_val > 1.0:
                is_anomaly = True
                dev_type = "PHYSICAL_BOUNDS"
                candidate_faults.extend(["dfa_sensor_falsification", "DFA_t21"])
                evidence.append(f"Motion confidence {conf_val} violates probability bounds [0.0, 1.0]")
                conf = 0.95

        elif self.sensor_id == "rfid" and payload.get("status") == "tampered":
            is_anomaly = True
            dev_type = "TAMPERING"
            candidate_faults.append("mitm_intercept_tamper")
            evidence.append("RFID payload tampering detected: status flagged as tampered")
            conf = 0.95

        # --------------------------------------------------------------
        # 3. Formal TFSM Conformance & State Machine Transition Check
        # --------------------------------------------------------------
        # Synchronize state if evaluating an isolated operational event
        if self.state == "s1" and event.get("observed_state") in ("s5", "s7", "s8", "s9"):
            self.state = str(event["observed_state"])

        inp_sym = self.reconstruct_input_symbol(event)
        tr = self.machine._on_input(self.state, inp_sym)

        if tr is None:
            exp_out = "o6"
            suspected_st = self.state
            suspected_tr = "undefined_transition"
            if obs_out != "o6":
                is_anomaly = True
                conformance = False
                if dev_type == "NONE":
                    dev_type = "OUTPUT_MISMATCH"
                evidence.append(f"State {self.state} received undefined input {inp_sym}: expected o6, observed {obs_out}")
        else:
            exp_out = tr.out
            suspected_st = self.state
            suspected_tr = tr.tid

            # Check Output Symbol Mismatch
            if obs_out and obs_out != exp_out:
                is_anomaly = True
                conformance = False

                # Check RFID Identity Spoofing (SPOOF_t23: unauthorized UID granted access)
                if self.sensor_id == "rfid" and tr.tid == "t23" and obs_out == "o11" and exp_out == "o12":
                    dev_type = "SPOOFING"
                    candidate_faults.extend(["rfid_uid_spoofing_t23", "SPOOF_t23"])
                    evidence.append("Identity Spoofing detected on rfid: unauthorized badge produced access-granted o11 output")
                    conf = 0.95
                else:
                    if dev_type == "NONE":
                        dev_type = "OUTPUT_MISMATCH"
                    evidence.append(f"Output mismatch on transition {tr.tid} from state {self.state}: expected {exp_out}, observed {obs_out}")
                    if obs_out == "o6":
                        # Only attribute DFA sensor fault if this is NOT a duplicate/replayed packet or flood
                        if dev_type not in ("REPLAY_DUPLICATE", "SEQUENCE_REGRESSION", "BURST_FLOOD"):
                            candidate_faults.extend([f"DFA_{tr.tid}", "dfa_sensor_falsification"])
                    elif obs_out == "o4" and tr.tid == "t9":
                        candidate_faults.append("RA_t9")

            # Advance nominal state machine ONLY if not a replayed duplicate
            if dev_type != "REPLAY_DUPLICATE":
                self.state = tr.dst

        # Update isolated monitor state
        self.last_timestamp = ts
        if dev_type != "REPLAY_DUPLICATE":
            self.last_seq = max(self.last_seq if self.last_seq is not None else seq, seq)
        else:
            if self.last_seq is None:
                self.last_seq = seq

        return EventDiagnosis(
            timestamp=ts,
            sensor_id=self.sensor_id,
            event_type=ev_type,
            sequence_number=seq,
            is_anomaly=is_anomaly,
            fsm_conformance=conformance,
            deviation_type=dev_type,
            suspected_state=suspected_st,
            suspected_transition=suspected_tr,
            expected_output=exp_out,
            observed_output=obs_out,
            candidate_faults=sorted(list(set(candidate_faults))),
            evidence=evidence,
            diagnosis_confidence=conf if is_anomaly else 1.0,
        )


# ----------------------------------------------------------------------
# Central Unified Diagnosis Engine
# ----------------------------------------------------------------------
class UnifiedDiagnosisEngine:
    """Central engine performing multi-sensor formal diagnosis and localization."""

    MODULES = {
        "motion": motion_sensor,
        "temperature": temperature_sensor,
        "ultrasonic": ultrasonic_sensor,
        "rfid": rfid_sensor,
    }

    def __init__(self) -> None:
        pass

    def diagnose_events(
        self,
        events: Union[List[Dict[str, Any]], List[Any]],
    ) -> SystemDiagnosisReport:
        """Analyzes an ordered sequence of observable events and generates diagnosis report.

        Guarantees:
        - Strict sensor isolation during evaluation.
        - NEVER reads 'is_attack', 'attack_type', 'attack_target', or 'fault_location'.
        """
        # Convert objects to observable dictionaries and strip quarantined ground-truth
        clean_events = []
        for e in events:
            if hasattr(e, "to_observable_dict"):
                clean_events.append(e.to_observable_dict())
            elif isinstance(e, dict):
                clean_e = copy.deepcopy(e)
                for qk in ("is_attack", "attack_type", "attack_target", "fault_location", "original_payload", "original_timestamp"):
                    clean_e.pop(qk, None)
                clean_events.append(clean_e)
            else:
                raise TypeError(f"Unsupported event type: {type(e)}")

        # Sort strictly by timestamp and sequence number
        clean_events.sort(key=lambda x: (float(x.get("timestamp", 0.0)), int(x.get("sequence_number", 0))))

        # Initialize independent monitors for each sensor
        monitors = {sid: SensorMonitor(sid, mod) for sid, mod in self.MODULES.items()}

        event_diagnoses: List[EventDiagnosis] = []
        events_by_sensor: Dict[str, List[EventDiagnosis]] = {sid: [] for sid in self.MODULES}

        # Evaluate events chronologically through isolated sensor monitors
        for ev in clean_events:
            sid = ev.get("sensor_id", "")
            if sid not in monitors:
                continue
            diag = monitors[sid].evaluate_event(ev)
            event_diagnoses.append(diag)
            events_by_sensor[sid].append(diag)

        # Build sensor diagnosis reports
        sensor_reports: Dict[str, SensorDiagnosisReport] = {}
        affected_sensors = []
        all_candidate_faults = []

        for sid, diag_list in events_by_sensor.items():
            anomalous = [d for d in diag_list if d.is_anomaly]
            has_anomaly = len(anomalous) > 0
            conformance = all(d.fsm_conformance for d in diag_list)

            if has_anomaly:
                affected_sensors.append(sid)

            sensor_faults = []
            sensor_evidence = []
            suspected_st = None
            suspected_tr = None
            exp_out = None
            obs_out = None
            conf_sum = 0.0

            for a in anomalous:
                sensor_faults.extend(a.candidate_faults)
                sensor_evidence.extend(a.evidence)
                if suspected_st is None and a.suspected_state:
                    suspected_st = a.suspected_state
                if suspected_tr is None and a.suspected_transition:
                    suspected_tr = a.suspected_transition
                if exp_out is None and a.expected_output:
                    exp_out = a.expected_output
                if obs_out is None and a.observed_output:
                    obs_out = a.observed_output
                conf_sum += a.diagnosis_confidence

            sensor_faults = sorted(list(set(sensor_faults)))
            all_candidate_faults.extend(sensor_faults)

            # Classify candidate attack category based strictly on isolated observable symptoms
            pred_attack = self._classify_attack_category(sid, sensor_faults, anomalous, monitors[sid])

            sensor_conf = round(conf_sum / len(anomalous), 2) if anomalous else 1.0

            sensor_reports[sid] = SensorDiagnosisReport(
                sensor_id=sid,
                anomaly_detected=has_anomaly,
                fsm_conformance=conformance,
                total_events=len(diag_list),
                anomalous_events_count=len(anomalous),
                predicted_attack_category=pred_attack,
                suspected_state=suspected_st,
                suspected_transition=suspected_tr,
                expected_output=exp_out,
                observed_output=obs_out,
                candidate_faults=sensor_faults,
                evidence=sensor_evidence[:8],
                diagnosis_confidence=sensor_conf,
            )

        affected_sensors = sorted(list(set(affected_sensors)))

        # Primary system-level predicted attack
        all_pred = [r.predicted_attack_category for r in sensor_reports.values() if r.predicted_attack_category != "Normal"]
        primary_attack = all_pred[0] if all_pred else "Normal"

        return SystemDiagnosisReport(
            overall_anomaly_detected=len(affected_sensors) > 0,
            affected_sensors=affected_sensors,
            sensor_reports=sensor_reports,
            primary_predicted_attack=primary_attack,
            event_diagnoses=event_diagnoses,
        )

    def _classify_attack_category(
        self,
        sensor_id: str,
        faults: List[str],
        anomalies: List[EventDiagnosis],
        monitor: SensorMonitor,
    ) -> str:
        """Classifies the most likely attack category from formal symptoms using evidence combinations."""
        if not anomalies:
            return "Normal"

        dev_types = {a.deviation_type for a in anomalies}

        # 1. RFID Identity Spoofing (SPOOF_t23 mutant or unauthorized access grant)
        if "SPOOFING" in dev_types or any("spoof" in f.lower() for f in faults):
            return "IdentitySpoofing"

        # Observable evidence tallies
        duplicate_seq_count = sum(1 for a in anomalies if a.deviation_type == "REPLAY_DUPLICATE")
        seq_regression_count = sum(1 for a in anomalies if a.deviation_type == "SEQUENCE_REGRESSION")
        burst_count = sum(1 for a in anomalies if a.deviation_type == "BURST_FLOOD")
        flood_frames = sum(1 for a in anomalies if a.event_type == "FLOOD")
        timing_violations = sum(1 for a in anomalies if a.deviation_type == "TIMING_VIOLATION")
        has_physical_bounds = "PHYSICAL_BOUNDS" in dev_types
        has_replay_fault = any("replay" in f.lower() or "ra_" in f.lower() for f in faults)
        has_delay_fault = any("delay" in f.lower() for f in faults)
        has_dfa_fault = any("dfa" in f.lower() for f in faults)
        has_dos_fault = any("dos" in f.lower() for f in faults)

        # 2. Genuine High-Rate DoS Flood (Flood frames or massive burst without duplicate sequences)
        if flood_frames >= 5 or (burst_count >= 5 and duplicate_seq_count == 0):
            return "DoS"

        # 3. Man-in-the-Middle (Payload tampering / semantic contradiction)
        if "TAMPERING" in dev_types or any("mitm" in f.lower() for f in faults):
            return "MITM"

        # 4. Replay Attack:
        # Observable evidence: duplicate sequence numbers already transmitted,
        # or sequence regressions without physical bounds violation or timeout
        if duplicate_seq_count > 0:
            return "Replay"

        if has_replay_fault and not has_physical_bounds and timing_violations == 0:
            return "Replay"

        # 5. Moderate DoS Flood (Burst rate on fresh sequence progression without replay duplicates)
        if (burst_count >= 2 or has_dos_fault) and duplicate_seq_count == 0:
            return "DoS"

        # 5. Data Falsification:
        # Observable evidence: payload violates physical domain bounds,
        # or sensor output symbol mismatch without replay or delay symptoms.
        if has_physical_bounds:
            return "DataFalsification"

        if (has_dfa_fault or "OUTPUT_MISMATCH" in dev_types) and duplicate_seq_count == 0 and timing_violations == 0:
            return "DataFalsification"

        # 6. Message Delay Attack:
        # Observable evidence: inter-arrival gap exceeds timeout threshold,
        # delayed arrival causing out-of-order sequence regression, but NO duplicate sequence numbers.
        if timing_violations > 0 or has_delay_fault:
            return "Delay"

        # 7. Remaining sequence regressions without timeout: Replay
        if seq_regression_count > 0 or has_replay_fault:
            return "Replay"

        if "OUTPUT_MISMATCH" in dev_types:
            return "DataFalsification"

        return "UnknownAnomaly"
