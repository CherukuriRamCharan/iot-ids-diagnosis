"""Unit tests for Stage 3 Four-Sensor Dataset Generation & Validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from features.build_unified_dataset import (
    GROUND_TRUTH_COLUMNS,
    METADATA_COLUMNS,
    OBSERVABLE_FEATURE_COLUMNS,
    generate_unified_dataset,
)


class TestStage3Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.tmp_path = Path(cls._tmp.name)
        # Generate a small 12-session test dataset in tmp
        cls.df, cls.manifest = generate_unified_dataset(
            n_sessions=24,
            session_duration=25.0,
            seed=42,
            out_dir=cls.tmp_path,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_all_four_sensors_represented(self) -> None:
        sensors = set(self.df["sensor_id"].unique())
        expected_sensors = {"motion", "temperature", "ultrasonic", "rfid"}
        self.assertEqual(sensors, expected_sensors)

        for s in expected_sensors:
            count = (self.df["sensor_id"] == s).sum()
            self.assertGreater(count, 50, f"Sensor {s} has too few rows: {count}")

    def test_all_seven_classes_present(self) -> None:
        classes = set(self.df["attack_type"].unique())
        expected_classes = {
            "Normal",
            "DoS",
            "Replay",
            "MITM",
            "DataFalsification",
            "Delay",
            "IdentitySpoofing",
        }
        self.assertEqual(classes, expected_classes)

    def test_splits_and_class_representation(self) -> None:
        splits = set(self.df["split"].unique())
        self.assertEqual(splits, {"train", "val", "test"})

        # Check that every split has Normal and all 6 attack classes
        expected_classes = {
            "Normal",
            "DoS",
            "Replay",
            "MITM",
            "DataFalsification",
            "Delay",
            "IdentitySpoofing",
        }
        for s in ("train", "val", "test"):
            split_classes = set(self.df[self.df["split"] == s]["attack_type"].unique())
            self.assertEqual(
                split_classes,
                expected_classes,
                f"Split '{s}' missing classes: {expected_classes - split_classes}",
            )

    def test_zero_leakage_in_observable_features(self) -> None:
        forbidden_quarantined = [
            "is_attack",
            "attack_type",
            "attack_target",
            "fault_location",
            "observed_state",
            "output_symbol",
            "original_payload",
            "original_timestamp",
            "session_id",
            "split",
        ]
        for col in forbidden_quarantined:
            self.assertNotIn(
                col,
                OBSERVABLE_FEATURE_COLUMNS,
                f"Leakage violation! Column '{col}' found in OBSERVABLE_FEATURE_COLUMNS",
            )

    def test_no_missing_or_infinite_values_in_x(self) -> None:
        X_df = self.df[list(OBSERVABLE_FEATURE_COLUMNS)]
        null_counts = X_df.isna().sum().to_dict()
        for col, count in null_counts.items():
            self.assertEqual(count, 0, f"Feature '{col}' contains {count} NaN values")

        inf_counts = np.isinf(X_df.values).sum()
        self.assertEqual(inf_counts, 0, "Feature matrix X contains infinite values")

    def test_session_level_splitting_disjointness(self) -> None:
        # Check that sessions are strictly disjoint across train, val, test
        train_sessions = set(self.df[self.df["split"] == "train"]["session_id"].unique())
        val_sessions = set(self.df[self.df["split"] == "val"]["session_id"].unique())
        test_sessions = set(self.df[self.df["split"] == "test"]["session_id"].unique())

        self.assertTrue(train_sessions.isdisjoint(val_sessions), "Train and Val share sessions!")
        self.assertTrue(train_sessions.isdisjoint(test_sessions), "Train and Test share sessions!")
        self.assertTrue(val_sessions.isdisjoint(test_sessions), "Val and Test share sessions!")

    def test_chronological_ordering_within_sessions(self) -> None:
        for sess_id in self.df["session_id"].unique():
            sess_df = self.df[self.df["session_id"] == sess_id]
            timestamps = sess_df["timestamp"].tolist()
            self.assertEqual(
                timestamps,
                sorted(timestamps),
                f"Session {sess_id} timestamps are not monotonic",
            )

    def test_label_consistency(self) -> None:
        # Normal rows must have is_attack=0
        normal_mask = self.df["attack_type"] == "Normal"
        self.assertTrue((self.df.loc[normal_mask, "is_attack"] == 0).all())
        self.assertTrue((self.df.loc[normal_mask, "attack_target"] == "None").all())
        self.assertTrue((self.df.loc[normal_mask, "fault_location"] == "None").all())

        # Attack rows must have is_attack=1
        attack_mask = self.df["attack_type"] != "Normal"
        self.assertTrue((self.df.loc[attack_mask, "is_attack"] == 1).all())
        self.assertTrue((self.df.loc[attack_mask, "attack_target"] != "None").all())
        self.assertTrue((self.df.loc[attack_mask, "fault_location"] != "None").all())

    def test_reproducibility_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_repr:
            df2, _ = generate_unified_dataset(
                n_sessions=4,
                session_duration=10.0,
                seed=42,
                out_dir=Path(tmp_repr),
            )
            df1, _ = generate_unified_dataset(
                n_sessions=4,
                session_duration=10.0,
                seed=42,
                out_dir=Path(tmp_repr),
            )
            self.assertEqual(len(df1), len(df2))
            pd.testing.assert_frame_equal(df1, df2)


if __name__ == "__main__":
    unittest.main()

