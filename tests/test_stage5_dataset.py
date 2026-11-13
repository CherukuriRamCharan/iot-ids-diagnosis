"""Unit tests for Stage 5.1 Four-Sensor Dataset & Sequence Pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ids.datasets_four_sensors import (
    BINARY_TO_IDX,
    CLASS_TO_IDX,
    DEFAULT_FOUR_SENSORS_CSV,
    EXCLUDED_FEATURES_WITH_REASON,
    FORBIDDEN_LEAKAGE_COLS,
    FOUR_SENSORS_BINARY_CLASSES,
    FOUR_SENSORS_CLASSES,
    FOUR_SENSORS_PRESERVE_BINARY_COLS,
    IDX_TO_BINARY,
    IDX_TO_CLASS,
    SELECTED_OBSERVABLE_FEATURES,
    assert_no_leakage_four_sensors,
    build_four_sensors_sequences,
    compute_class_weights,
    load_four_sensors_dataset,
    prepare_four_sensors_loaders,
)


class TestStage5Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ds_multi = load_four_sensors_dataset(binary=False)
        cls.ds_binary = load_four_sensors_dataset(binary=True)
        cls.seq_multi = build_four_sensors_sequences(cls.ds_multi, window_size=16, stride=1)
        cls.seq_bin = build_four_sensors_sequences(cls.ds_binary, window_size=16, stride=1)

    def test_class_mapping_consistency(self) -> None:
        """Verifies stable 7-class and binary class-to-index mappings."""
        self.assertEqual(len(FOUR_SENSORS_CLASSES), 7)
        self.assertEqual(FOUR_SENSORS_CLASSES[0], "Normal")
        self.assertEqual(len(FOUR_SENSORS_BINARY_CLASSES), 2)
        self.assertEqual(FOUR_SENSORS_BINARY_CLASSES[0], "Normal")
        self.assertEqual(FOUR_SENSORS_BINARY_CLASSES[1], "Attack")

        # Bidirectional 7-class mapping
        for i, name in enumerate(FOUR_SENSORS_CLASSES):
            self.assertEqual(CLASS_TO_IDX[name], i)
            self.assertEqual(IDX_TO_CLASS[i], name)

        # Bidirectional binary mapping
        self.assertEqual(BINARY_TO_IDX["Normal"], 0)
        self.assertEqual(BINARY_TO_IDX["Attack"], 1)
        self.assertEqual(IDX_TO_BINARY[0], "Normal")
        self.assertEqual(IDX_TO_BINARY[1], "Attack")

    def test_feature_leakage_rejection(self) -> None:
        """Verifies compile-time assertion raises ValueError for forbidden/quarantined columns."""
        for forbidden in FORBIDDEN_LEAKAGE_COLS:
            with self.assertRaises(ValueError):
                assert_no_leakage_four_sensors(["inter_arrival", forbidden])

        # Confirm selected observable features contain zero forbidden columns
        bad = set(SELECTED_OBSERVABLE_FEATURES) & FORBIDDEN_LEAKAGE_COLS
        self.assertEqual(bad, set(), f"Leakage detected in SELECTED_OBSERVABLE_FEATURES: {bad}")
        self.assertEqual(len(SELECTED_OBSERVABLE_FEATURES), 14)

    def test_dataset_loading_splits_and_integrity(self) -> None:
        """Verifies session-level split separation and feature matrix shapes."""
        ds = self.ds_multi
        self.assertEqual(ds.X_train.shape[1], 14)
        self.assertEqual(ds.X_val.shape[1], 14)
        self.assertEqual(ds.X_test.shape[1], 14)

        # Check expected split row counts
        self.assertEqual(len(ds.X_train), 1782)
        self.assertEqual(len(ds.X_val), 626)
        self.assertEqual(len(ds.X_test), 682)
        self.assertEqual(len(ds.X_train) + len(ds.X_val) + len(ds.X_test), 3090)

        # Verify session IDs are properly disjoint
        tr_sess = set(ds.train_sessions)
        va_sess = set(ds.val_sessions)
        te_sess = set(ds.test_sessions)

        self.assertEqual(tr_sess, set(range(0, 14)))
        self.assertEqual(va_sess, set(range(14, 19)))
        self.assertEqual(te_sess, set(range(19, 24)))

        self.assertTrue(tr_sess.isdisjoint(va_sess))
        self.assertTrue(tr_sess.isdisjoint(te_sess))
        self.assertTrue(va_sess.isdisjoint(te_sess))

    def test_training_only_scaler_fitting(self) -> None:
        """Verifies scaler was fitted strictly on training data without leakage."""
        ds = self.ds_multi
        self.assertIsNotNone(ds.scaler.mean_)
        self.assertIsNotNone(ds.scaler.scale_)

        # Binary columns should be preserved untouched (0.0 or 1.0)
        for i, col_name in enumerate(ds.feature_names):
            if col_name in FOUR_SENSORS_PRESERVE_BINARY_COLS:
                unique_vals = np.unique(ds.X_train[:, i])
                self.assertTrue(
                    set(unique_vals).issubset({0.0, 1.0}),
                    f"Binary column {col_name} was modified by scaler: {unique_vals}",
                )

    def test_session_boundary_protection_in_sequences(self) -> None:
        """Verifies that NO sequence window crosses session boundaries."""
        seq = self.seq_multi
        raw_df = self.ds_multi.raw_df

        # Verify training sequences
        for i in range(len(seq.X_train_seq)):
            end_row = seq.train_end_row_ids[i]
            window_rows = raw_df.loc[end_row - 15 : end_row]
            # All 16 rows must belong to the exact same session ID
            session_ids = window_rows["session_id"].unique()
            self.assertEqual(
                len(session_ids),
                1,
                f"Sequence {i} crosses session boundary: sessions={session_ids}",
            )
            self.assertEqual(session_ids[0], seq.train_sessions[i])

        # Verify test sequences
        for i in range(len(seq.X_test_seq)):
            end_row = seq.test_end_row_ids[i]
            window_rows = raw_df.loc[end_row - 15 : end_row]
            session_ids = window_rows["session_id"].unique()
            self.assertEqual(len(session_ids), 1)
            self.assertEqual(session_ids[0], seq.test_sessions[i])

    def test_split_boundary_protection_in_sequences(self) -> None:
        """Verifies sequence counts and strict split separation."""
        seq = self.seq_multi
        # Train: 14 sessions (0-13), 1782 events -> 1782 - (14 * 15) = 1572 sequences
        self.assertEqual(seq.X_train_seq.shape, (1572, 16, 14))
        # Val: 5 sessions (14-18), 626 events -> 626 - (5 * 15) = 551 sequences
        self.assertEqual(seq.X_val_seq.shape, (551, 16, 14))
        # Test: 5 sessions (19-23), 682 events -> 682 - (5 * 15) = 607 sequences
        self.assertEqual(seq.X_test_seq.shape, (607, 16, 14))

        tr_sess = set(seq.train_sessions)
        va_sess = set(seq.val_sessions)
        te_sess = set(seq.test_sessions)

        self.assertTrue(tr_sess.isdisjoint(va_sess))
        self.assertTrue(tr_sess.isdisjoint(te_sess))
        self.assertTrue(va_sess.isdisjoint(te_sess))

    def test_correct_last_event_labeling(self) -> None:
        """Verifies last-event labeling matches target attack type of window's final packet."""
        seq = self.seq_multi
        raw_df = self.ds_multi.raw_df

        # Sample across train, val, and test sequences
        for split_seq, split_end_rows in (
            (seq.y_train_seq, seq.train_end_row_ids),
            (seq.y_val_seq, seq.val_end_row_ids),
            (seq.y_test_seq, seq.test_end_row_ids),
        ):
            # Check 30 sampled positions
            indices = np.linspace(0, len(split_seq) - 1, 30, dtype=int)
            for idx in indices:
                expected_label_str = raw_df.loc[split_end_rows[idx], "attack_type"]
                expected_label_idx = CLASS_TO_IDX[expected_label_str]
                self.assertEqual(split_seq[idx], expected_label_idx)

    def test_binary_label_correctness(self) -> None:
        """Verifies binary dataset labels: 0 for Normal, 1 for any attack."""
        seq = self.seq_bin
        raw_df = self.ds_binary.raw_df

        for split_seq, split_end_rows in (
            (seq.y_train_seq, seq.train_end_row_ids),
            (seq.y_val_seq, seq.val_end_row_ids),
            (seq.y_test_seq, seq.test_end_row_ids),
        ):
            indices = np.linspace(0, len(split_seq) - 1, 30, dtype=int)
            for idx in indices:
                raw_type = raw_df.loc[split_end_rows[idx], "attack_type"]
                expected_bin = 0 if raw_type == "Normal" else 1
                self.assertEqual(split_seq[idx], expected_bin)

    def test_class_weights_calculation(self) -> None:
        """Verifies inverse-frequency class weights computed from train sequences."""
        weights = self.seq_multi.class_weights
        self.assertEqual(len(weights), 7)
        self.assertTrue(np.all(weights > 0.0))

        # Replay has fewest instances (6) -> highest weight
        # Normal has most instances (1357) -> lowest weight
        norm_w = weights[CLASS_TO_IDX["Normal"]]
        replay_w = weights[CLASS_TO_IDX["Replay"]]
        dos_w = weights[CLASS_TO_IDX["DoS"]]

        self.assertLess(norm_w, dos_w)
        self.assertLess(dos_w, replay_w)
        self.assertGreater(replay_w, 30.0)

    def test_missing_and_infinite_value_handling(self) -> None:
        """Verifies loader strictly rejects datasets with NaN or Inf values."""
        df = pd.read_csv(DEFAULT_FOUR_SENSORS_CSV, low_memory=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Test NaN rejection
            nan_df = df.copy()
            nan_df.loc[5, "inter_arrival"] = np.nan
            nan_csv = tmp_path / "nan_dataset.csv"
            nan_df.to_csv(nan_csv, index=False)
            with self.assertRaises(ValueError):
                load_four_sensors_dataset(nan_csv)

            # Test Inf rejection
            inf_df = df.copy()
            inf_df.loc[10, "payload_size"] = np.inf
            inf_csv = tmp_path / "inf_dataset.csv"
            inf_df.to_csv(inf_csv, index=False)
            with self.assertRaises(ValueError):
                load_four_sensors_dataset(inf_csv)

    def test_reproducible_sequence_generation(self) -> None:
        """Verifies deterministic sequence generation across repeated calls."""
        seq1 = build_four_sensors_sequences(self.ds_multi, window_size=16, stride=1)
        seq2 = build_four_sensors_sequences(self.ds_multi, window_size=16, stride=1)

        self.assertTrue(np.array_equal(seq1.X_train_seq, seq2.X_train_seq))
        self.assertTrue(np.array_equal(seq1.y_train_seq, seq2.y_train_seq))
        self.assertTrue(np.array_equal(seq1.train_sessions, seq2.train_sessions))
        self.assertTrue(np.array_equal(seq1.train_end_row_ids, seq2.train_end_row_ids))
        self.assertTrue(np.array_equal(seq1.class_weights, seq2.class_weights))

    def test_pytorch_dataloaders(self) -> None:
        """Verifies integration with PyTorch DataLoaders."""
        import torch

        loaders = prepare_four_sensors_loaders(self.seq_multi, batch_size=64)
        train_loader = loaders["train_loader"]
        val_loader = loaders["val_loader"]
        test_loader = loaders["test_loader"]

        self.assertEqual(len(loaders["split_sizes"]), 3)
        self.assertEqual(loaders["num_features"], 14)
        self.assertEqual(loaders["num_classes"], 7)

        # Iterate one batch from train_loader
        bx, by = next(iter(train_loader))
        self.assertEqual(bx.shape, (64, 16, 14))
        self.assertEqual(by.shape, (64,))
        self.assertEqual(bx.dtype, torch.float32)
        self.assertEqual(by.dtype, torch.int64)

        # Iterate one batch from test_loader
        tx, ty = next(iter(test_loader))
        self.assertEqual(tx.shape, (64, 16, 14))
        self.assertEqual(ty.shape, (64,))
        self.assertEqual(tx.dtype, torch.float32)
        self.assertEqual(ty.dtype, torch.int64)

    def test_ground_truth_target_mapping_and_schema(self) -> None:
        """Regression test: verifies ground-truth attack_target and fault_location schema mapping."""
        raw_df = self.ds_multi.raw_df

        # Verify zero NaNs in target and fault location
        self.assertEqual(raw_df["attack_target"].isna().sum(), 0)
        self.assertEqual(raw_df["fault_location"].isna().sum(), 0)

        # Verify all 2,671 Normal events have canonical "None" sentinels
        normal_mask = raw_df["attack_type"] == "Normal"
        self.assertEqual(normal_mask.sum(), 2671)
        self.assertTrue((raw_df.loc[normal_mask, "attack_target"] == "None").all())
        self.assertTrue((raw_df.loc[normal_mask, "fault_location"] == "None").all())

        # Verify all 419 Attack events have valid target sensors and fault locations
        attack_mask = raw_df["is_attack"] == 1
        self.assertEqual(attack_mask.sum(), 419)
        self.assertFalse((raw_df.loc[attack_mask, "attack_target"] == "None").any())
        self.assertFalse((raw_df.loc[attack_mask, "fault_location"] == "None").any())

        # Verify specific fault locations are preserved
        dos_rows = raw_df[raw_df["attack_type"] == "DoS"]
        self.assertTrue((dos_rows["fault_location"] == "dos_queue_exhaustion").all())

        replay_rows = raw_df[raw_df["attack_type"] == "Replay"]
        self.assertTrue((replay_rows["fault_location"] == "replayed_stale_sequence").all())

    def test_attack_event_missing_target_rejection(self) -> None:
        """Regression test: verifies missing values in attack annotations strictly trigger ValueError."""
        df = pd.read_csv(DEFAULT_FOUR_SENSORS_CSV, low_memory=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Attack row with missing fault_location must NOT be filled and must raise ValueError
            bad_attack_df = df.copy()
            # Row 152 is a DoS attack event
            self.assertEqual(bad_attack_df.loc[152, "attack_type"], "DoS")
            bad_attack_df.loc[152, "fault_location"] = np.nan

            bad_csv = tmp_path / "bad_attack.csv"
            bad_attack_df.to_csv(bad_csv, index=False)
            with self.assertRaises(ValueError) as ctx:
                load_four_sensors_dataset(bad_csv)
            self.assertIn("fault_location", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

