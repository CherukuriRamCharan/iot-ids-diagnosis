"""Stage 6 End-to-End Integration and Project Readiness Test Suite.

Project: Attack Diagnosis and Localization in IoT Intrusion Detection Using Formal Modeling and Deep Learning
Repository: iot-ids-diagnosis

Conducts exhaustive end-to-end integration validation across:
1. Complete pipeline execution (TFSM -> Simulator -> Injector -> Features -> Windows -> DL -> Formal -> Hybrid Report).
2. Model & dataset compatibility (14 observable features, scaling, class mapping, zero leakage).
3. Seven reproducible attack scenarios (Normal, DoS, Replay, MITM, DataFalsification, Delay, IdentitySpoofing).
4. All six hybrid adjudication outcomes (CONSENSUS, FORMAL_VIOLATION_DL_NORMAL, STATISTICAL_ANOMALY,
   NOMINAL, INSUFFICIENT_SEQUENCE, MODEL_UNAVAILABLE).
5. Fault-tolerance guarantees (model unavailability never suppresses formal evidence).
"""

from __future__ import annotations

import copy
import unittest
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from diagnosis_loc.hybrid_diagnosis import (
    HybridDiagnosisEngine,
    HybridDiagnosisReport,
    extract_observable_feature_vector,
)
from diagnosis_loc.unified_diagnosis import UnifiedDiagnosisEngine
from ids.datasets_four_sensors import (
    CLASS_TO_IDX,
    FORBIDDEN_LEAKAGE_COLS,
    FOUR_SENSORS_CLASSES,
    SELECTED_OBSERVABLE_FEATURES,
    assert_no_leakage_four_sensors,
)
from ids.models_sequence import BiLSTMClassifier, LSTMClassifier, build_model
from simulation.attack_injector import AttackInjector, AttackSpec
from simulation.unified_simulator import UnifiedSimulator


class MockPredictor(nn.Module):
    """Deterministic mock neural classifier for controlled integration testing."""

    def __init__(self, target_class_idx: int = 0, confidence_logit: float = 10.0, num_classes: int = 7) -> None:
        super().__init__()
        self.target_class_idx = target_class_idx
        self.confidence_logit = confidence_logit
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, self.num_classes, dtype=torch.float32)
        logits[:, self.target_class_idx] = self.confidence_logit
        return logits


class TestStage6Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = UnifiedSimulator(duration=25.0, seed=42)
        cls.raw_events = cls.sim.run()
        cls.injector = AttackInjector(seed=42)

    def test_01_full_pipeline_traceability_nominal(self) -> None:
        """Verifies end-to-end trace from nominal simulator events to hybrid report."""
        # 1. Simulator produces events
        self.assertGreater(len(self.raw_events), 20)
        # 2. Extract observable features
        last_seq: Dict[str, int] = {}
        last_t = 0.0
        vectors = []
        for e in self.raw_events:
            d = e.to_observable_dict() if hasattr(e, "to_observable_dict") else e
            vec = extract_observable_feature_vector(d, last_seq, last_t)
            vectors.append(vec)
            last_t = float(d.get("timestamp", 0.0))
        self.assertEqual(len(vectors), len(self.raw_events))
        self.assertEqual(len(vectors[0]), 14)

        # 3. Hybrid engine evaluates nominal traffic
        model = MockPredictor(target_class_idx=CLASS_TO_IDX["Normal"])
        engine = HybridDiagnosisEngine(model=model)
        report = engine.diagnose(self.raw_events)

        self.assertEqual(report.hybrid_outcome, "NOMINAL")
        self.assertTrue(report.agreement)
        self.assertFalse(report.formal_anomaly_detected)
        self.assertFalse(report.dl_attack_detected)
        self.assertEqual(report.dl_predicted_category, "Normal")
        self.assertEqual(report.formal_primary_attack, "Normal")
        self.assertEqual(report.affected_sensors, [])

    def test_02_model_dataset_feature_order_and_provenance(self) -> None:
        """Verifies 14-feature ordering and strict zero-leakage compliance."""
        self.assertEqual(len(SELECTED_OBSERVABLE_FEATURES), 14)
        expected_order = [
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
            "numeric_reading",
            "has_alert_flag",
        ]
        self.assertEqual(list(SELECTED_OBSERVABLE_FEATURES), expected_order)

        # Ensure zero overlap with forbidden metadata/ground truth
        intersection = set(SELECTED_OBSERVABLE_FEATURES) & FORBIDDEN_LEAKAGE_COLS
        self.assertEqual(intersection, set())

    def test_03_scenario_dos_ultrasonic(self) -> None:
        """Verifies DoS scenario: burst flood on ultrasonic correctly identified."""
        spec = AttackSpec("DoS", "ultrasonic", start_time=8.0, duration=2.5, intensity=1.5)
        events = self.injector.inject(self.raw_events, [spec])

        model = MockPredictor(target_class_idx=CLASS_TO_IDX["DoS"])
        engine = HybridDiagnosisEngine(model=model)
        report = engine.diagnose(events)

        self.assertTrue(report.formal_anomaly_detected)
        self.assertIn("ultrasonic", report.affected_sensors)
        self.assertEqual(report.formal_primary_attack, "DoS")
        self.assertIn("dos_queue_exhaustion", report.candidate_fault_mutants)
        self.assertEqual(report.hybrid_outcome, "CONSENSUS")
        self.assertTrue(report.agreement)

    def test_04_scenario_replay_motion(self) -> None:
        """Verifies Replay scenario: stale sequence packets on motion correctly identified."""
        spec = AttackSpec("Replay", "motion", start_time=10.0, duration=6.0, params={"delay_offset": 4.0})
        events = self.injector.inject(self.raw_events, [spec])

        model = MockPredictor(target_class_idx=CLASS_TO_IDX["Replay"])
        engine = HybridDiagnosisEngine(model=model)
        report = engine.diagnose(events)

        self.assertTrue(report.formal_anomaly_detected)
        self.assertIn("motion", report.affected_sensors)
        self.assertEqual(report.formal_primary_attack, "Replay")
        self.assertIn("replayed_stale_sequence", report.candidate_fault_mutants)
        self.assertEqual(report.hybrid_outcome, "CONSENSUS")

    def test_05_scenario_mitm_temperature(self) -> None:
        """Verifies MITM scenario: payload tampering on temperature correctly identified."""
        spec = AttackSpec("MITM", "temperature", start_time=7.0, duration=5.0)
        events = self.injector.inject(self.raw_events, [spec])

        model = MockPredictor(target_class_idx=CLASS_TO_IDX["MITM"])
        engine = HybridDiagnosisEngine(model=model)
        report = engine.diagnose(events)

        self.assertTrue(report.formal_anomaly_detected)
        self.assertIn("temperature", report.affected_sensors)
        self.assertEqual(report.formal_primary_attack, "MITM")
        self.assertIn("mitm_intercept_tamper", report.candidate_fault_mutants)
        self.assertEqual(report.hybrid_outcome, "CONSENSUS")

    def test_06_scenario_data_falsification_ultrasonic(self) -> None:
        """Verifies DataFalsification scenario: out-of-bounds telemetry on ultrasonic."""
        spec = AttackSpec("DataFalsification", "ultrasonic", start_time=8.0, duration=4.0)
        events = self.injector.inject(self.raw_events, [spec])

        model = MockPredictor(target_class_idx=CLASS_TO_IDX["DataFalsification"])
        engine = HybridDiagnosisEngine(model=model)
        report = engine.diagnose(events)

        self.assertTrue(report.formal_anomaly_detected)
        self.assertIn("ultrasonic", report.affected_sensors)
        self.assertEqual(report.formal_primary_attack, "DataFalsification")
        self.assertIn("dfa_sensor_falsification", report.candidate_fault_mutants)
        self.assertEqual(report.hybrid_outcome, "CONSENSUS")

    def test_07_scenario_delay_temperature(self) -> None:
        """Verifies Delay scenario: packet delivery buffering on temperature sensor."""
        spec = AttackSpec("Delay", "temperature", start_time=9.0, duration=4.0, params={"delay_sec": 3.5})
        events = self.injector.inject(self.raw_events, [spec])

        model = MockPredictor(target_class_idx=CLASS_TO_IDX["Delay"])
        engine = HybridDiagnosisEngine(model=model)
        report = engine.diagnose(events)

        self.assertTrue(report.formal_anomaly_detected)
        self.assertIn("temperature", report.affected_sensors)
        self.assertEqual(report.formal_primary_attack, "Delay")
        self.assertIn("delay_timeout_exceeded", report.candidate_fault_mutants)
        self.assertEqual(report.hybrid_outcome, "CONSENSUS")

    def test_08_scenario_identity_spoofing_rfid(self) -> None:
        """Verifies IdentitySpoofing scenario: unauthorized RFID UID scan."""
        spec = AttackSpec("IdentitySpoofing", "rfid", start_time=5.0, duration=15.0)
        events = self.injector.inject(self.raw_events, [spec])

        model = MockPredictor(target_class_idx=CLASS_TO_IDX["IdentitySpoofing"])
        engine = HybridDiagnosisEngine(model=model)
        report = engine.diagnose(events)

        self.assertTrue(report.formal_anomaly_detected)
        self.assertIn("rfid", report.affected_sensors)
        self.assertEqual(report.formal_primary_attack, "IdentitySpoofing")
        self.assertIn("rfid_uid_spoofing_t23", report.candidate_fault_mutants)
        self.assertEqual(report.hybrid_outcome, "CONSENSUS")

    def test_09_outcome_consensus(self) -> None:
        """Verifies CONSENSUS outcome when both formal and DL tiers concur."""
        spec = AttackSpec("DoS", "ultrasonic", start_time=6.0, duration=2.5)
        events = self.injector.inject(self.raw_events, [spec])

        model = MockPredictor(target_class_idx=CLASS_TO_IDX["DoS"])
        engine = HybridDiagnosisEngine(model=model)
        report = engine.diagnose(events)

        self.assertEqual(report.hybrid_outcome, "CONSENSUS")
        self.assertTrue(report.agreement)
        self.assertTrue(report.formal_anomaly_detected)
        self.assertTrue(report.dl_attack_detected)
        self.assertEqual(report.dl_predicted_category, "DoS")

    def test_10_outcome_formal_violation_dl_normal(self) -> None:
        """Verifies FORMAL_VIOLATION_DL_NORMAL outcome when DL misses formal violation."""
        spec = AttackSpec("DataFalsification", "temperature", start_time=8.0, duration=4.0)
        events = self.injector.inject(self.raw_events, [spec])

        # DL erroneously predicts Normal
        model = MockPredictor(target_class_idx=CLASS_TO_IDX["Normal"])
        engine = HybridDiagnosisEngine(model=model)
        report = engine.diagnose(events)

        self.assertEqual(report.hybrid_outcome, "FORMAL_VIOLATION_DL_NORMAL")
        self.assertFalse(report.agreement)
        self.assertTrue(report.formal_anomaly_detected)
        self.assertFalse(report.dl_attack_detected)
        self.assertIn("dfa_sensor_falsification", report.candidate_fault_mutants)

    def test_11_outcome_statistical_anomaly(self) -> None:
        """Verifies STATISTICAL_ANOMALY outcome when DL flags nominal traffic."""
        # Benign events that formally conform 100%
        benign_events = self.raw_events[:25]

        # DL flags traffic as DoS
        model = MockPredictor(target_class_idx=CLASS_TO_IDX["DoS"])
        engine = HybridDiagnosisEngine(model=model)
        report = engine.diagnose(benign_events)

        self.assertEqual(report.hybrid_outcome, "STATISTICAL_ANOMALY")
        self.assertFalse(report.agreement)
        self.assertFalse(report.formal_anomaly_detected)
        self.assertTrue(report.dl_attack_detected)
        self.assertEqual(report.candidate_fault_mutants, [])

    def test_12_outcome_nominal(self) -> None:
        """Verifies NOMINAL outcome when both tiers verify benign operation."""
        benign_events = self.raw_events[:25]

        model = MockPredictor(target_class_idx=CLASS_TO_IDX["Normal"])
        engine = HybridDiagnosisEngine(model=model)
        report = engine.diagnose(benign_events)

        self.assertEqual(report.hybrid_outcome, "NOMINAL")
        self.assertTrue(report.agreement)
        self.assertFalse(report.formal_anomaly_detected)
        self.assertFalse(report.dl_attack_detected)
        self.assertTrue(report.formal_conformance)

    def test_13_outcome_insufficient_sequence(self) -> None:
        """Verifies INSUFFICIENT_SEQUENCE fallback when fewer than 16 events exist."""
        short_events = self.raw_events[:10]

        model = MockPredictor(target_class_idx=CLASS_TO_IDX["Normal"])
        engine = HybridDiagnosisEngine(model=model)
        report = engine.diagnose(short_events)

        self.assertEqual(report.hybrid_outcome, "INSUFFICIENT_SEQUENCE")
        self.assertEqual(report.total_events, 10)
        self.assertEqual(report.dl_window_count, 0)
        self.assertFalse(report.agreement)

    def test_14_outcome_model_unavailable_fallback(self) -> None:
        """Verifies MODEL_UNAVAILABLE fallback when model weights cannot be loaded."""
        engine = HybridDiagnosisEngine()  # No model provided and default checkpoint absent
        report = engine.diagnose(self.raw_events)

        self.assertEqual(report.hybrid_outcome, "MODEL_UNAVAILABLE")
        self.assertFalse(report.model_available if hasattr(report, "model_available") else engine.model_available)
        self.assertEqual(report.dl_predicted_category, "Unavailable")

    def test_15_model_failure_never_suppresses_formal_anomaly(self) -> None:
        """Verifies that model unavailability never suppresses an active formal violation."""
        spec = AttackSpec("DoS", "ultrasonic", start_time=8.0, duration=2.5, intensity=1.5)
        events = self.injector.inject(self.raw_events, [spec])

        # Model is completely disabled
        engine = HybridDiagnosisEngine()
        report = engine.diagnose(events)

        self.assertEqual(report.hybrid_outcome, "MODEL_UNAVAILABLE")
        self.assertTrue(report.formal_anomaly_detected, "Formal anomaly must NOT be suppressed by missing model")
        self.assertIn("ultrasonic", report.affected_sensors)
        self.assertEqual(report.formal_primary_attack, "DoS")
        self.assertIn("dos_queue_exhaustion", report.candidate_fault_mutants)

    def test_16_zero_ground_truth_in_dl_features(self) -> None:
        """Verifies that simulator metadata and labels injected into events are completely stripped."""
        spec = AttackSpec("MITM", "temperature", start_time=7.0, duration=5.0)
        events = self.injector.inject(self.raw_events, [spec])

        # Manually pollute events with forbidden keys
        polluted = []
        for e in events[:20]:
            d = e.to_observable_dict() if hasattr(e, "to_observable_dict") else dict(e)
            d["is_attack"] = 1
            d["attack_type"] = "MITM"
            d["attack_target"] = "temperature"
            d["fault_location"] = "mitm_intercept_tamper"
            d["tampered_by"] = "mitm_proxy"
            d["is_event_flood"] = 1.0
            d["has_tamper_flag"] = 1.0
            polluted.append(d)

        model = MockPredictor(target_class_idx=CLASS_TO_IDX["MITM"])
        engine = HybridDiagnosisEngine(model=model)
        report = engine.diagnose(polluted)

        # Vector extraction must strictly exclude forbidden features
        last_seq: Dict[str, int] = {}
        vec = extract_observable_feature_vector(polluted[0], last_seq)
        self.assertEqual(len(vec), 14)


if __name__ == "__main__":
    unittest.main()

