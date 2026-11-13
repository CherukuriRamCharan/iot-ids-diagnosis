"""Unit tests for Stage 5.3 Hybrid Deep Learning + Formal TFSM Diagnosis Engine."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from diagnosis_loc.hybrid_diagnosis import (
    HybridDiagnosisEngine,
    HybridDiagnosisReport,
    extract_observable_feature_vector,
)
from ids.datasets_four_sensors import (
    CLASS_TO_IDX,
    FOUR_SENSORS_CLASSES,
    SELECTED_OBSERVABLE_FEATURES,
)
from ids.models_sequence import BiLSTMClassifier, build_model
from simulation.attack_injector import AttackInjector, AttackSpec
from simulation.unified_simulator import UnifiedSimulator


class MockClassifier(nn.Module):
    """Mock PyTorch classifier emitting deterministic class probabilities for unit testing."""

    def __init__(self, target_class_idx: int = 0, confidence: float = 0.95, num_classes: int = 7) -> None:
        super().__init__()
        self.target_class_idx = target_class_idx
        self.confidence = confidence
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, self.num_classes, dtype=torch.float32)
        # Set large logit for target class to yield desired confidence
        logits[:, self.target_class_idx] = 10.0
        return logits


class TestStage5Hybrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sim = UnifiedSimulator(duration=25.0, seed=42)
        cls.raw_events = cls.sim.run()
        cls.injector = AttackInjector(seed=42)

    def test_1_successful_model_loading(self) -> None:
        """Verifies checkpoint weights and metadata load correctly into the engine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pt_path = tmp / "model.pt"
            json_path = tmp / "model.json"

            # Create lightweight dummy checkpoint
            model = build_model("BiLSTM", num_features=14, hidden_size=64, num_classes=7)
            torch.save(
                {
                    "model_name": "BiLSTM",
                    "task_type": "multiclass",
                    "seed": 42,
                    "state_dict": model.state_dict(),
                    "feature_names": list(SELECTED_OBSERVABLE_FEATURES),
                    "class_names": list(FOUR_SENSORS_CLASSES),
                    "config": {"num_features": 14, "hidden_size": 64, "num_layers": 2, "num_classes": 7, "dropout": 0.2},
                },
                pt_path,
            )
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "meta": {
                            "model_name": "BiLSTM",
                            "seed": 42,
                            "scaler_params": {"mean": [0.0] * 14, "scale": [1.0] * 14},
                        }
                    },
                    f,
                )

            engine = HybridDiagnosisEngine(checkpoint_path=pt_path, metadata_path=json_path)
            self.assertTrue(engine.model_available)
            self.assertIsNotNone(engine.model)
            self.assertEqual(len(engine.scaler_mean), 14)

    def test_2_model_input_feature_ordering(self) -> None:
        """Verifies feature ordering matches the 14 approved observable features exactly."""
        engine = HybridDiagnosisEngine(model=MockClassifier())
        self.assertEqual(engine.feature_names, SELECTED_OBSERVABLE_FEATURES)
        self.assertEqual(len(engine.feature_names), 14)

        # Verify ordering of specific critical features
        self.assertEqual(engine.feature_names[0], "inter_arrival")
        self.assertEqual(engine.feature_names[1], "payload_size")
        self.assertEqual(engine.feature_names[2], "qos")
        self.assertEqual(engine.feature_names[3], "seq_delta")
        self.assertEqual(engine.feature_names[12], "numeric_reading")
        self.assertEqual(engine.feature_names[13], "has_alert_flag")

    def test_3_correct_16_event_window_construction(self) -> None:
        """Verifies sliding window count equals N - 16 + 1."""
        engine = HybridDiagnosisEngine(model=MockClassifier(target_class_idx=0))
        # Take exactly 20 events
        subset = self.raw_events[:20]
        report = engine.diagnose(subset)

        self.assertEqual(report.total_events, 20)
        self.assertEqual(report.dl_window_count, 5)  # 20 - 16 + 1 = 5
        self.assertEqual(len(report.dl_window_predictions), 5)

    def test_4_session_and_sensor_isolation(self) -> None:
        """Verifies sensor and session boundaries remain isolated during feature extraction."""
        engine = HybridDiagnosisEngine(model=MockClassifier(target_class_idx=0))
        subset = self.raw_events[:25]
        report = engine.diagnose(subset)

        # Confirm all 4 sensors are tracked by formal monitor
        self.assertIn("motion", report.system_diagnosis_report.sensor_reports)
        self.assertIn("temperature", report.system_diagnosis_report.sensor_reports)
        self.assertIn("ultrasonic", report.system_diagnosis_report.sensor_reports)
        self.assertIn("rfid", report.system_diagnosis_report.sensor_reports)

    def test_5_consensus_outcome(self) -> None:
        """Verifies CONSENSUS when formal FSM detects anomaly and DL predicts attack."""
        # Inject DoS flood attack on motion
        spec = AttackSpec("DoS", "motion", start_time=6.0, duration=3.0, intensity=1.5)
        attack_events = self.injector.inject(self.raw_events, [spec])

        dos_idx = CLASS_TO_IDX["DoS"]
        mock_model = MockClassifier(target_class_idx=dos_idx, confidence=0.98)
        engine = HybridDiagnosisEngine(model=mock_model)

        report = engine.diagnose(attack_events)

        self.assertEqual(report.hybrid_outcome, "CONSENSUS")
        self.assertTrue(report.agreement)
        self.assertTrue(report.formal_anomaly_detected)
        self.assertTrue(report.dl_attack_detected)
        self.assertEqual(report.dl_predicted_category, "DoS")
        self.assertEqual(report.formal_primary_attack, "DoS")
        self.assertIn("dos_queue_exhaustion", report.candidate_fault_mutants)
        self.assertGreater(len(report.formal_evidence), 0)

    def test_6_formal_violation_with_dl_predicting_normal(self) -> None:
        """Verifies FORMAL_VIOLATION_DL_NORMAL when formal detects violation but DL predicts Normal."""
        spec = AttackSpec("DoS", "motion", start_time=6.0, duration=3.0, intensity=1.5)
        attack_events = self.injector.inject(self.raw_events, [spec])

        normal_idx = CLASS_TO_IDX["Normal"]
        mock_model = MockClassifier(target_class_idx=normal_idx, confidence=0.95)
        engine = HybridDiagnosisEngine(model=mock_model)

        report = engine.diagnose(attack_events)

        self.assertEqual(report.hybrid_outcome, "FORMAL_VIOLATION_DL_NORMAL")
        self.assertFalse(report.agreement)
        self.assertTrue(report.formal_anomaly_detected)
        self.assertFalse(report.dl_attack_detected)
        self.assertEqual(report.dl_predicted_category, "Normal")
        # Mathematical formal violation proof is preserved!
        self.assertIn("dos_queue_exhaustion", report.candidate_fault_mutants)
        self.assertIn("FORMAL VIOLATION", report.outcome_explanation)

    def test_7_statistical_anomaly_with_formal_conformance_passing(self) -> None:
        """Verifies STATISTICAL_ANOMALY when formal passes but DL predicts attack."""
        # Pure benign event stream
        benign_events = self.raw_events[:30]

        dos_idx = CLASS_TO_IDX["DoS"]
        mock_model = MockClassifier(target_class_idx=dos_idx, confidence=0.92)
        engine = HybridDiagnosisEngine(model=mock_model)

        report = engine.diagnose(benign_events)

        self.assertEqual(report.hybrid_outcome, "STATISTICAL_ANOMALY")
        self.assertFalse(report.agreement)
        self.assertFalse(report.formal_anomaly_detected)
        self.assertTrue(report.formal_conformance)
        self.assertTrue(report.dl_attack_detected)
        self.assertEqual(report.dl_predicted_category, "DoS")
        # Formal evidence must remain empty - no false proof created!
        self.assertEqual(report.candidate_fault_mutants, [])
        self.assertIn("STATISTICAL ANOMALY", report.outcome_explanation)

    def test_8_nominal_outcome(self) -> None:
        """Verifies NOMINAL when both formal passes and DL predicts Normal."""
        benign_events = self.raw_events[:30]

        normal_idx = CLASS_TO_IDX["Normal"]
        mock_model = MockClassifier(target_class_idx=normal_idx, confidence=0.99)
        engine = HybridDiagnosisEngine(model=mock_model)

        report = engine.diagnose(benign_events)

        self.assertEqual(report.hybrid_outcome, "NOMINAL")
        self.assertTrue(report.agreement)
        self.assertFalse(report.formal_anomaly_detected)
        self.assertTrue(report.formal_conformance)
        self.assertFalse(report.dl_attack_detected)
        self.assertEqual(report.dl_predicted_category, "Normal")
        self.assertEqual(report.candidate_fault_mutants, [])

    def test_9_insufficient_sequence_handling(self) -> None:
        """Verifies INSUFFICIENT_SEQUENCE when event stream has fewer than 16 events."""
        short_events = self.raw_events[:10]  # Only 10 events (< 16)
        engine = HybridDiagnosisEngine(model=MockClassifier())

        report = engine.diagnose(short_events)

        self.assertEqual(report.hybrid_outcome, "INSUFFICIENT_SEQUENCE")
        self.assertEqual(report.dl_window_count, 0)
        self.assertEqual(report.dl_predicted_category, "Unknown")
        self.assertFalse(report.agreement)
        self.assertIn("Insufficient sequence length", report.outcome_explanation)

    def test_10_model_unavailable_handling(self) -> None:
        """Verifies MODEL_UNAVAILABLE does not suppress formal anomaly."""
        spec = AttackSpec("DoS", "motion", start_time=6.0, duration=3.0, intensity=1.5)
        attack_events = self.injector.inject(self.raw_events, [spec])

        # Engine initialized with non-existent checkpoint
        engine = HybridDiagnosisEngine(checkpoint_path="nonexistent_checkpoint.pt")
        self.assertFalse(engine.model_available)

        report = engine.diagnose(attack_events)

        self.assertEqual(report.hybrid_outcome, "MODEL_UNAVAILABLE")
        # Formal anomaly is NOT suppressed!
        self.assertTrue(report.formal_anomaly_detected)
        self.assertEqual(report.formal_primary_attack, "DoS")
        self.assertIn("dos_queue_exhaustion", report.candidate_fault_mutants)
        self.assertEqual(report.dl_predicted_category, "Unavailable")

    def test_11_formal_evidence_preservation(self) -> None:
        """Verifies candidate faults, deviation type, and formal evidence are preserved exactly."""
        spec = AttackSpec("Delay", "ultrasonic", start_time=6.0, duration=4.0, params={"delay_sec": 3.5})
        attack_events = self.injector.inject(self.raw_events, [spec])

        engine = HybridDiagnosisEngine(model=MockClassifier(target_class_idx=CLASS_TO_IDX["Delay"]))
        report = engine.diagnose(attack_events)

        self.assertEqual(report.hybrid_outcome, "CONSENSUS")
        self.assertIn("delay_timeout_exceeded", report.candidate_fault_mutants)
        self.assertEqual(report.deviation_type, "TIMING_VIOLATION")
        self.assertGreater(len(report.formal_evidence), 0)
        self.assertIn("ultrasonic", report.affected_sensors)

    def test_12_no_ground_truth_metadata_leakage(self) -> None:
        """Verifies that simulator metadata and labels are stripped before feature extraction."""
        polluted_events = []
        for e in self.raw_events[:20]:
            d = e.to_observable_dict()
            # Inject forbidden ground-truth metadata
            d["is_attack"] = 1
            d["attack_type"] = "DoS"
            d["tampered_by"] = "mitm_proxy"
            d["is_event_flood"] = 1.0
            d["has_tamper_flag"] = 1.0
            d["fault_location"] = "dos_queue_exhaustion"
            polluted_events.append(d)

        engine = HybridDiagnosisEngine(model=MockClassifier())
        report = engine.diagnose(polluted_events)

        # Verify feature vector length is strictly 14 and matches observable definition
        last_seq = {}
        vec = extract_observable_feature_vector(polluted_events[0], last_seq)
        self.assertEqual(len(vec), 14)

    def test_13_no_fabricated_observed_fsm_outputs(self) -> None:
        """Verifies that DL predictions never synthesize or alter formal expected/observed outputs."""
        benign_events = self.raw_events[:20]
        # DL predicts extreme attack with 1.0 confidence
        engine = HybridDiagnosisEngine(model=MockClassifier(target_class_idx=CLASS_TO_IDX["DataFalsification"]))
        report = engine.diagnose(benign_events)

        # Formal engine must remain completely un-fabricated
        self.assertTrue(report.formal_conformance)
        self.assertEqual(report.candidate_fault_mutants, [])
        self.assertIsNone(report.deviation_type)

    def test_14_deterministic_inference(self) -> None:
        """Verifies deterministic identical output on repeated diagnosis calls."""
        engine = HybridDiagnosisEngine(model=MockClassifier(target_class_idx=0))
        subset = self.raw_events[:25]

        rep1 = engine.diagnose(subset)
        rep2 = engine.diagnose(subset)

        self.assertEqual(rep1.hybrid_outcome, rep2.hybrid_outcome)
        self.assertEqual(rep1.dl_confidence_score, rep2.dl_confidence_score)
        self.assertEqual(rep1.formal_anomaly_detected, rep2.formal_anomaly_detected)
        self.assertEqual(rep1.candidate_fault_mutants, rep2.candidate_fault_mutants)

    def test_15_hybrid_diagnosis_report_dataclass_instantiation(self) -> None:
        """Regression test: verifies HybridDiagnosisReport dataclass field ordering and instantiation."""
        # 1. Instantiate with required fields only (verifying default factories and default values)
        rep_required = HybridDiagnosisReport(
            hybrid_outcome="NOMINAL",
            outcome_explanation="Nominal operation verified.",
            agreement=True,
            total_events=20,
            dl_attack_detected=False,
            dl_predicted_category="Normal",
            dl_confidence_score=0.985,
            dl_window_count=5,
            formal_anomaly_detected=False,
            formal_conformance=True,
            formal_primary_attack="Normal",
            affected_sensors=[],
        )
        self.assertEqual(rep_required.hybrid_outcome, "NOMINAL")
        self.assertEqual(rep_required.dl_window_predictions, [])
        self.assertEqual(rep_required.candidate_fault_mutants, [])
        self.assertEqual(rep_required.formal_evidence, [])
        self.assertIsNone(rep_required.suspected_sensor)
        self.assertIsNone(rep_required.suspected_transition)
        self.assertIsNone(rep_required.suspected_state)
        self.assertIsNone(rep_required.expected_output)
        self.assertIsNone(rep_required.observed_output)
        self.assertIsNone(rep_required.deviation_type)
        self.assertIsNone(rep_required.system_diagnosis_report)

        # 2. Instantiate with both required and optional defaulted fields
        rep_full = HybridDiagnosisReport(
            hybrid_outcome="CONSENSUS",
            outcome_explanation="Consensus attack detected.",
            agreement=True,
            total_events=25,
            dl_attack_detected=True,
            dl_predicted_category="DoS",
            dl_confidence_score=0.992,
            dl_window_count=10,
            formal_anomaly_detected=True,
            formal_conformance=False,
            formal_primary_attack="DoS",
            affected_sensors=["ultrasonic"],
            dl_window_predictions=[{"window_index": 0, "predicted_class": "DoS"}],
            suspected_sensor="ultrasonic",
            suspected_transition="t21",
            suspected_state="s9",
            expected_output="o11",
            observed_output="o6",
            deviation_type="TIMING_VIOLATION",
            candidate_fault_mutants=["DFA_t21", "BD_s8"],
            formal_evidence=["BURST_FLOOD"],
        )
        self.assertEqual(rep_full.hybrid_outcome, "CONSENSUS")
        self.assertEqual(len(rep_full.dl_window_predictions), 1)
        self.assertEqual(rep_full.suspected_sensor, "ultrasonic")
        self.assertEqual(rep_full.candidate_fault_mutants, ["DFA_t21", "BD_s8"])
        self.assertEqual(rep_full.formal_evidence, ["BURST_FLOOD"])

        # 3. Verify positional instantiation of the 12 required fields
        rep_pos = HybridDiagnosisReport(
            "STATISTICAL_ANOMALY",
            "Statistical anomaly explanation",
            False,
            18,
            True,
            "Replay",
            0.875,
            3,
            False,
            True,
            "Normal",
            [],
        )
        self.assertEqual(rep_pos.hybrid_outcome, "STATISTICAL_ANOMALY")
        self.assertEqual(rep_pos.dl_predicted_category, "Replay")
        self.assertEqual(rep_pos.dl_window_predictions, [])


if __name__ == "__main__":
    unittest.main()


