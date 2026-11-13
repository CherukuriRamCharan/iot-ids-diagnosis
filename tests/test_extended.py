"""
Extended unit tests for:
  1. Temperature EFSM state transitions and detectors (existing)
  2. Motion EFSM state transitions and detectors (new)
  3. Diagnosis rules (rolling window, payload evidence, signal logic)
  4. Dataset schema validation
  5. Model loading (RF binary)
  6. Sequence generation sanity checks

Run from any directory:
  .venv\\Scripts\\python.exe -m pytest tests\\test_extended.py -v
"""

import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ids"))
sys.path.insert(0, str(PROJECT_ROOT / "models_fsm"))

from diagnosis_rules import (
    changed_fields, compare_message_pairs, diagnosis_signals,
    payload_evidence, rate_exceeded, rolling_window_counts,
)
from temperature_efsm import (
    TemperatureEFSM,
    INIT, CONNECTED, GENERATE_DATA, PUBLISH_DATA, WAIT_TIMER,
    DISCONNECTED, ANOMALOUS,
)
from motion_efsm import MotionEFSM
from motion_efsm import SENSE


# =============================================================================
# 1. Temperature EFSM
# =============================================================================

class TemperatureEFSMTests(unittest.TestCase):

    def test_normal_lifecycle_reaches_wait_timer(self):
        m = TemperatureEFSM()
        self.assertEqual(m.state, INIT)
        m.connect()
        self.assertEqual(m.state, CONNECTED)
        m.timer_expired()
        self.assertEqual(m.state, GENERATE_DATA)
        m.data_generated(25.0)
        self.assertEqual(m.state, PUBLISH_DATA)
        m.publish_success()
        self.assertEqual(m.state, WAIT_TIMER)

    def test_disconnect_from_any_active_state(self):
        for start_state in (CONNECTED, WAIT_TIMER, GENERATE_DATA, PUBLISH_DATA):
            m = TemperatureEFSM()
            m.state = start_state
            m.disconnect()
            self.assertEqual(m.state, DISCONNECTED)

    def test_invalid_transition_does_not_change_state(self):
        m = TemperatureEFSM()
        # timer_expired while in INIT is not a valid transition
        result = m.timer_expired()
        self.assertEqual(m.state, INIT)
        self.assertIn("Invalid", result)

    def test_replay_detector_flags_repeated_payload(self):
        m = TemperatureEFSM()
        m.detect_replay(22.0)   # first: no previous
        result = m.detect_replay(22.0)  # duplicate
        self.assertIn("DETECTED", result)
        self.assertEqual(m.state, ANOMALOUS)

    def test_data_falsification_flags_out_of_range(self):
        m = TemperatureEFSM(min_temperature=-10, max_temperature=60)
        ok = m.detect_data_falsification(25.0)
        self.assertNotIn("DETECTED", ok)
        self.assertNotEqual(m.state, ANOMALOUS)
        result = m.detect_data_falsification(95.0)
        self.assertIn("DETECTED", result)
        self.assertEqual(m.state, ANOMALOUS)

    def test_boundary_values_are_valid(self):
        m = TemperatureEFSM(min_temperature=-10, max_temperature=60)
        self.assertNotIn("DETECTED", m.detect_data_falsification(-10))
        m2 = TemperatureEFSM(min_temperature=-10, max_temperature=60)
        self.assertNotIn("DETECTED", m2.detect_data_falsification(60))

    def test_message_manipulation_flags_value_change(self):
        m = TemperatureEFSM()
        result = m.detect_message_manipulation(25.0, 40.0)
        self.assertIn("DETECTED", result)
        self.assertEqual(m.state, ANOMALOUS)

    def test_no_manipulation_when_values_equal(self):
        m = TemperatureEFSM()
        result = m.detect_message_manipulation(25.0, 25.0)
        self.assertNotIn("DETECTED", result)

    def test_excessive_communication_triggers_above_threshold(self):
        m = TemperatureEFSM(max_messages_per_window=3, window_seconds=10)
        for ts in [1.0, 2.0, 3.0]:
            m.detect_excessive_communication(timestamp=ts)
        result = m.detect_excessive_communication(timestamp=4.0)
        self.assertIn("DETECTED", result)
        self.assertEqual(m.state, ANOMALOUS)

    def test_excessive_communication_does_not_trigger_below_threshold(self):
        m = TemperatureEFSM(max_messages_per_window=5, window_seconds=10)
        for ts in [1.0, 2.0, 3.0]:
            result = m.detect_excessive_communication(timestamp=ts)
            self.assertNotIn("DETECTED", result)


# =============================================================================
# 2. Motion EFSM
# =============================================================================

class MotionEFSMTests(unittest.TestCase):

    def test_normal_lifecycle_reaches_wait_timer(self):
        m = MotionEFSM()
        m.connect()
        m.timer_expired()
        self.assertEqual(m.state, SENSE)
        m.motion_sensed(1)
        self.assertEqual(m.state, PUBLISH_DATA)
        m.publish_success()
        self.assertEqual(m.state, WAIT_TIMER)

    def test_invalid_motion_value_triggers_falsification(self):
        m = MotionEFSM()
        result = m.detect_invalid_value(2)
        self.assertIn("DETECTED", result)
        self.assertEqual(m.state, ANOMALOUS)

    def test_valid_motion_values_pass(self):
        m = MotionEFSM()
        for v in (0, 1):
            result = m.detect_invalid_value(v)
            self.assertNotIn("DETECTED", result)

    def test_repeated_motion_value_is_evidence_only(self):
        """
        Consecutive identical binary readings are evidence only;
        the EFSM should not enter ANOMALOUS for a single repeat.
        """
        m = MotionEFSM()
        m.detect_replay(0)
        result = m.detect_replay(0)
        self.assertIn("EVIDENCE", result)
        # Should NOT enter ANOMALOUS for evidence-only
        self.assertNotEqual(m.state, ANOMALOUS)

    def test_message_manipulation_triggers_for_binary_values(self):
        m = MotionEFSM()
        result = m.detect_message_manipulation(original_value=1, forwarded_value=0)
        self.assertIn("DETECTED", result)
        self.assertEqual(m.state, ANOMALOUS)

    def test_no_manipulation_when_values_equal(self):
        m = MotionEFSM()
        result = m.detect_message_manipulation(original_value=0, forwarded_value=0)
        self.assertNotIn("DETECTED", result)

    def test_excessive_communication_fires_above_threshold(self):
        m = MotionEFSM(max_messages_per_window=3, window_seconds=10)
        for ts in [1.0, 2.0, 3.0]:
            m.detect_excessive_communication(timestamp=ts)
        result = m.detect_excessive_communication(timestamp=4.0)
        self.assertIn("DETECTED", result)

    def test_disconnect_clears_to_disconnected_state(self):
        from motion_efsm import DISCONNECTED
        m = MotionEFSM()
        m.connect()
        m.disconnect()
        self.assertEqual(m.state, DISCONNECTED)


# =============================================================================
# 3. Diagnosis rules (already covered in test_diagnosis_rules.py; add new)
# =============================================================================

class DiagnosisRuleExtendedTests(unittest.TestCase):

    def test_all_three_signals_present(self):
        signals = diagnosis_signals(
            range_violations=1,
            rate_violations=1,
            changed_pairs=1,
        )
        self.assertEqual(len(signals), 3)
        self.assertIn("temperature_range_violation", signals)
        self.assertIn("excessive_publish_rate", signals)
        self.assertIn("paired_original_forwarded_value_change", signals)

    def test_no_signals_when_all_zero(self):
        self.assertEqual(diagnosis_signals(0, 0, 0), [])

    def test_rate_exceeded_boundary(self):
        self.assertTrue(rate_exceeded(6, 5))
        self.assertFalse(rate_exceeded(5, 5))
        self.assertFalse(rate_exceeded(4, 5))

    def test_rolling_window_empty_input(self):
        self.assertEqual(rolling_window_counts([], window_seconds=10), [])

    def test_rolling_window_single_packet(self):
        self.assertEqual(rolling_window_counts([100.0], window_seconds=10), [1])

    def test_payload_evidence_skips_non_json_rows(self):
        evidence = payload_evidence([
            {"decode_status": "decode_error", "temperature": 99},
            {"decode_status": "decoded_json",  "temperature": 25,
             "payload_text": '{"temperature":25}'},
        ])
        self.assertEqual(evidence["decoded_json_rows"], 1)
        self.assertEqual(evidence["out_of_range_rows"], [])


# =============================================================================
# 4. Dataset schema validation
# =============================================================================

class DatasetSchemaTests(unittest.TestCase):

    ALL_FEATURES_CSV = (
        PROJECT_ROOT / "data" / "base_paper" / "all_packet_features.csv"
    )
    EXPECTED_COLUMNS = {
        "capture_file", "label", "timestamp", "packet_length",
        "tcp_src_port", "tcp_dst_port", "udp_src_port", "udp_dst_port",
        "tcp_syn", "tcp_ack", "tcp_reset", "udp_length",
    }

    def test_csv_exists(self):
        self.assertTrue(
            self.ALL_FEATURES_CSV.exists(),
            f"Dataset CSV not found: {self.ALL_FEATURES_CSV}",
        )

    def test_csv_has_expected_columns(self):
        if not self.ALL_FEATURES_CSV.exists():
            self.skipTest("Dataset CSV not available")
        header = pd.read_csv(self.ALL_FEATURES_CSV, nrows=0).columns.tolist()
        missing = self.EXPECTED_COLUMNS - set(header)
        self.assertFalse(
            missing,
            f"Missing expected columns: {sorted(missing)}",
        )

    def test_csv_label_classes(self):
        if not self.ALL_FEATURES_CSV.exists():
            self.skipTest("Dataset CSV not available")
        # Read only the label column to check classes.
        labels = pd.read_csv(
            self.ALL_FEATURES_CSV,
            usecols=["label"],
            low_memory=False,
        )["label"].str.lower().unique()
        expected = {"benign", "dos", "mirai", "mitm", "scan"}
        actual   = set(labels)
        self.assertTrue(
            expected.issubset(actual),
            f"Expected classes {expected}, found {actual}",
        )

    def test_mqtt_traffic_features_schema(self):
        csv = PROJECT_ROOT / "features" / "mqtt_traffic_features.csv"
        if not csv.exists():
            self.skipTest("mqtt_traffic_features.csv not generated yet")
        df = pd.read_csv(csv, nrows=0)
        for col in ("capture_file", "traffic_label", "timestamp_epoch",
                    "mqtt_message_type", "mqtt_topic"):
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_mqtt_payload_features_schema(self):
        csv = PROJECT_ROOT / "features" / "mqtt_payload_features.csv"
        if not csv.exists():
            self.skipTest("mqtt_payload_features.csv not generated yet")
        df = pd.read_csv(csv, nrows=0)
        for col in ("capture_file", "timestamp_epoch", "mqtt_topic",
                    "temperature", "motion", "payload_text", "decode_status"):
            self.assertIn(col, df.columns, f"Missing column: {col}")


# =============================================================================
# 5. Model loading
# =============================================================================

class ModelLoadingTests(unittest.TestCase):

    RF_PATH = PROJECT_ROOT / "models" / "base_paper_rf.joblib"

    def test_binary_rf_model_loads(self):
        if not self.RF_PATH.exists():
            self.skipTest("base_paper_rf.joblib not found — train first")
        import joblib
        bundle = joblib.load(self.RF_PATH)
        self.assertIn("model", bundle)
        self.assertIn("features", bundle)
        self.assertIsInstance(bundle["features"], list)

    def test_binary_rf_predicts_expected_classes(self):
        if not self.RF_PATH.exists():
            self.skipTest("base_paper_rf.joblib not found — train first")
        import joblib
        bundle = joblib.load(self.RF_PATH)
        model  = bundle["model"]
        feats  = bundle["features"]
        dummy  = pd.DataFrame([[0] * len(feats)], columns=feats)
        pred   = model.predict(dummy)
        self.assertIn(pred[0], ("benign", "attack"))


# =============================================================================
# 6. Sequence generation sanity check
# =============================================================================

class SequenceGenerationTests(unittest.TestCase):

    def _make_capture_df(self, n_rows=50, label="benign"):
        """Create a minimal fake capture DataFrame."""
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "capture_file":  [f"{label}-fake.pcap"] * n_rows,
            "label":         [label] * n_rows,
            "timestamp":     np.sort(rng.uniform(0, 100, n_rows)),
            "packet_length": rng.integers(40, 1500, n_rows),
            "iat_seconds":   np.concatenate([[0.0], rng.uniform(0, 5, n_rows - 1)]),
            "rolling_10s_pkt_rate": rng.integers(1, 20, n_rows),
            "tcp_syn":       rng.integers(0, 2, n_rows),
            "tcp_ack":       rng.integers(0, 2, n_rows),
            "tcp_reset":     rng.integers(0, 2, n_rows),
            "udp_length":    rng.integers(0, 500, n_rows),
        })

    def test_sequence_count_for_single_capture(self):
        """With n_steps=5 and 50 rows, expect 50-5+1 = 46 sequences."""
        sys.path.insert(0, str(PROJECT_ROOT / "ids"))
        from train_lstm import build_sequences, FEATURES_IAT
        df = self._make_capture_df(n_rows=50)
        df["label_5"] = "benign"
        X, y, caps = build_sequences(df, FEATURES_IAT, n_steps=5,
                                     max_per_capture=1000)
        self.assertEqual(len(X), 46)
        self.assertTrue(all(c == "benign-fake.pcap" for c in caps))

    def test_sequences_have_correct_shape(self):
        sys.path.insert(0, str(PROJECT_ROOT / "ids"))
        from train_lstm import build_sequences, FEATURES_IAT
        df = self._make_capture_df(n_rows=50)
        df["label_5"] = "benign"
        X, _, _ = build_sequences(df, FEATURES_IAT, n_steps=5,
                                  max_per_capture=1000)
        self.assertEqual(X.shape, (46, 5, len(FEATURES_IAT)))

    def test_captures_below_n_steps_are_skipped(self):
        sys.path.insert(0, str(PROJECT_ROOT / "ids"))
        from train_lstm import build_sequences, FEATURES_IAT
        df = self._make_capture_df(n_rows=3)
        df["label_5"] = "benign"
        with self.assertRaises(ValueError):
            build_sequences(df, FEATURES_IAT, n_steps=5, max_per_capture=1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
