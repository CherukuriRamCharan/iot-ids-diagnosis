"""Unit tests for Stage 3A leakage-safe dataset loader, masked normalizer,
and within-split sequence window builder (ids/datasets_real.py).

Uses small synthetic fixtures so tests run quickly without reading large CSVs.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ids.datasets_real import (
    MQTT_UAD_CLASSES,
    MQTTSET_CLASSES,
    MQTTSET_PRESERVE_BINARY_COLS,
    UAD_PRESERVE_BINARY_COLS,
    MaskedStandardScaler,
    assert_no_leakage_columns,
    build_mqtt_uad_sequences,
    compute_class_weights,
    load_mqtt_uad,
    load_mqttset,
)


def _create_synthetic_uad(root: Path) -> None:
    cols = [
        "source_file",
        "source_row_idx",
        "frame.time_delta",
        "frame.len",
        "has_ip",
        "has_mqtt",
        "has_mqtt_topic",
        "has_mqtt_msg",
        "mqtt.msgtype",
        "mqtt.len",
        "mqtt.all_missing",
        "mqtt.const_val",
        "type",
        "split",
    ]
    rows = [
        # DoS.csv: 4 train, 3 val, 2 test (all >= W=3 except test < 3)
        ("DoS.csv", 0, 0.1, 60, 1, 0, 0, 0, -1.0, -1.0, -1.0, 5.0, "normal", "train"),
        ("DoS.csv", 1, 0.2, 80, 1, 1, 1, 1, 3.0, 20.0, -1.0, 5.0, "DoS", "train"),
        ("DoS.csv", 2, 0.3, 100, 1, 1, 1, 1, 3.0, 40.0, -1.0, 5.0, "DoS", "train"),
        ("DoS.csv", 3, 0.4, 120, 1, 1, 1, 1, 3.0, 60.0, -1.0, 5.0, "DoS", "train"),
        ("DoS.csv", 4, 0.5, 70, 1, 0, 0, 0, -1.0, -1.0, -1.0, 5.0, "normal", "val"),
        ("DoS.csv", 5, 0.6, 90, 1, 1, 1, 1, 1.0, 30.0, -1.0, 5.0, "DoS", "val"),
        ("DoS.csv", 6, 0.7, 110, 1, 1, 1, 1, 3.0, 50.0, -1.0, 5.0, "DoS", "val"),
        ("DoS.csv", 7, 0.8, 65, 1, 0, 0, 0, -1.0, -1.0, -1.0, 5.0, "normal", "test"),
        ("DoS.csv", 8, 0.9, 95, 1, 1, 1, 1, 3.0, 35.0, -1.0, 5.0, "DoS", "test"),
        # MitM.csv: 3 train, 2 val, 3 test
        ("MitM.csv", 0, 0.05, 42, 0, 0, 0, 0, -1.0, -1.0, -1.0, 5.0, "mitm", "train"),
        ("MitM.csv", 1, 0.15, 42, 0, 0, 0, 0, -1.0, -1.0, -1.0, 5.0, "mitm", "train"),
        ("MitM.csv", 2, 0.25, 60, 1, 1, 1, 1, 3.0, 15.0, -1.0, 5.0, "normal", "train"),
        ("MitM.csv", 3, 0.35, 42, 0, 0, 0, 0, -1.0, -1.0, -1.0, 5.0, "mitm", "val"),
        ("MitM.csv", 4, 0.45, 60, 1, 0, 0, 0, -1.0, -1.0, -1.0, 5.0, "normal", "val"),
        ("MitM.csv", 5, 0.55, 42, 0, 0, 0, 0, -1.0, -1.0, -1.0, 5.0, "mitm", "test"),
        ("MitM.csv", 6, 0.65, 60, 1, 1, 0, 0, 13.0, 2.0, -1.0, 5.0, "intrusion", "test"),
        ("MitM.csv", 7, 0.75, 60, 1, 1, 1, 1, 3.0, 18.0, -1.0, 5.0, "normal", "test"),
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(root / "mqtt_uad_features.csv", index=False)

    meta = df[["source_file", "source_row_idx", "type", "split"]].copy()
    meta["ip.src"] = "192.168.1.10"
    meta["mqtt.topic"] = "distance/ultrasonic1"
    meta.to_csv(root / "mqtt_uad_metadata.csv", index=False)

    feat_cols = [
        "frame.time_delta",
        "frame.len",
        "has_ip",
        "has_mqtt",
        "has_mqtt_topic",
        "has_mqtt_msg",
        "mqtt.msgtype",
        "mqtt.len",
        "mqtt.all_missing",
        "mqtt.const_val",
    ]
    manifest = {"feature_columns": feat_cols}
    (root / "mqtt_uad_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _create_synthetic_mqttset(root: Path) -> None:
    cols = [
        "source_row_idx",
        "Flow_Duration",
        "Tot_Fwd_Pkts",
        "has_inf_rate",
        "label",
        "Cat",
        "split",
    ]
    rows = [
        (0, 100.0, 2.0, 0, "Normal", "Normal", "train"),
        (1, 200.0, 4.0, 0, "Anomaly", "Flood", "train"),
        (2, 300.0, 6.0, 1, "Anomaly", "Slowite", "train"),
        (5, 150.0, 3.0, 0, "Normal", "Normal", "val"),
        (6, 250.0, 5.0, 0, "Anomaly", "Bruteforce", "val"),
        (9, 180.0, 2.0, 0, "Anomaly", "Malformed", "test"),
        (10, 220.0, 4.0, 0, "Anomaly", "MalariaDoS", "test"),
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(root / "mqttset_features.csv", index=False)

    meta = pd.DataFrame(
        {
            "source_row_idx": df["source_row_idx"],
            "Flow_ID": [f"f_{i}" for i in range(len(df))],
            "Src_IP": ["192.168.1.90"] * len(df),
            "Protocol": [6] * len(df),
            "label": df["label"],
            "Cat": df["Cat"],
            "split": df["split"],
        }
    )
    meta.to_csv(root / "mqttset_metadata.csv", index=False)

    manifest = {"feature_columns": ["Flow_Duration", "Tot_Fwd_Pkts", "has_inf_rate"]}
    (root / "mqttset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class TestPhase3Data(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.uad_dir = self.tmp_dir / "mqtt_uad"
        self.set_dir = self.tmp_dir / "mqttset"
        self.uad_dir.mkdir()
        self.set_dir.mkdir()
        _create_synthetic_uad(self.uad_dir)
        _create_synthetic_mqttset(self.set_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_1_dataset_loading_and_shapes(self) -> None:
        uad = load_mqtt_uad(self.uad_dir)
        self.assertEqual(uad.X_train.shape, (7, 10))
        self.assertEqual(uad.X_val.shape, (5, 10))
        self.assertEqual(uad.X_test.shape, (5, 10))

        uad_mqtt = load_mqtt_uad(self.uad_dir, mqtt_only=True)
        self.assertEqual(uad_mqtt.X_train.shape[0], 4)
        self.assertTrue(np.all(uad_mqtt.X_train[:, uad_mqtt.feature_names.index("has_mqtt")] == 1))

        mset = load_mqttset(self.set_dir)
        self.assertEqual(mset.X_train.shape, (3, 3))
        self.assertEqual(mset.X_val.shape, (2, 3))
        self.assertEqual(mset.X_test.shape, (2, 3))

    def test_2_label_preservation(self) -> None:
        uad_mc = load_mqtt_uad(self.uad_dir, binary=False)
        self.assertEqual(uad_mc.class_names, MQTT_UAD_CLASSES)
        self.assertEqual(uad_mc.y_train.tolist(), [0, 1, 1, 1, 2, 2, 0])
        self.assertEqual(uad_mc.y_test.tolist(), [0, 1, 2, 3, 0])

        uad_bin = load_mqtt_uad(self.uad_dir, binary=True)
        self.assertEqual(uad_bin.y_test.tolist(), [0, 1, 1, 1, 0])

        mset_mc = load_mqttset(self.set_dir, binary=False)
        self.assertEqual(mset_mc.class_names, MQTTSET_CLASSES)
        self.assertEqual(mset_mc.y_train.tolist(), [0, 1, 5])
        self.assertEqual(mset_mc.y_test.tolist(), [4, 2])

    def test_3_metadata_exclusion_from_features(self) -> None:
        uad = load_mqtt_uad(self.uad_dir)
        mset = load_mqttset(self.set_dir)
        assert_no_leakage_columns(uad.feature_names)
        assert_no_leakage_columns(mset.feature_names)
        with self.assertRaises(ValueError):
            assert_no_leakage_columns(["frame.len", "ip.src"])
        with self.assertRaises(ValueError):
            assert_no_leakage_columns(["Flow_Duration", "Protocol"])

    def test_4_split_integrity(self) -> None:
        uad = load_mqtt_uad(self.uad_dir)
        tr_pairs = set(zip(uad.train_groups.tolist(), uad.train_row_ids.tolist()))
        va_pairs = set(zip(uad.val_groups.tolist(), uad.val_row_ids.tolist()))
        te_pairs = set(zip(uad.test_groups.tolist(), uad.test_row_ids.tolist()))
        self.assertEqual(len(tr_pairs & va_pairs), 0)
        self.assertEqual(len(tr_pairs & te_pairs), 0)
        self.assertEqual(len(va_pairs & te_pairs), 0)

        mset = load_mqttset(self.set_dir)
        self.assertEqual(len(set(mset.train_groups) & set(mset.val_groups)), 0)
        self.assertEqual(len(set(mset.train_groups) & set(mset.test_groups)), 0)
        self.assertEqual(len(set(mset.val_groups) & set(mset.test_groups)), 0)

    def test_5_train_only_normalization(self) -> None:
        uad = load_mqtt_uad(self.uad_dir)
        scaler1 = MaskedStandardScaler(
            uad.feature_names, sentinel_value=-1.0, preserve_cols=UAD_PRESERVE_BINARY_COLS
        ).fit(uad.X_train)
        X_tr1 = scaler1.transform(uad.X_train)

        # Corrupt val and test with extreme values; fit(X_train) must be identical
        X_val_extreme = uad.X_val * 1000.0
        scaler2 = MaskedStandardScaler(
            uad.feature_names, sentinel_value=-1.0, preserve_cols=UAD_PRESERVE_BINARY_COLS
        ).fit(uad.X_train)
        _ = scaler2.transform(X_val_extreme)
        np.testing.assert_allclose(scaler1.mean_, scaler2.mean_)
        np.testing.assert_allclose(scaler1.scale_, scaler2.scale_)
        np.testing.assert_allclose(X_tr1, scaler2.transform(uad.X_train))

    def test_6_sentinel_masking_and_binary_preservation(self) -> None:
        uad = load_mqtt_uad(self.uad_dir)
        scaler = MaskedStandardScaler(
            uad.feature_names, sentinel_value=-1.0, preserve_cols=UAD_PRESERVE_BINARY_COLS
        )
        X_tr = scaler.fit_transform(uad.X_train)

        # Sentinel -1.0 entries in mqtt.len (idx 7) must become 0.0 and not affect mean_
        len_idx = uad.feature_names.index("mqtt.len")
        raw_valid = uad.X_train[:, len_idx][uad.X_train[:, len_idx] != -1.0]
        self.assertAlmostEqual(float(scaler.mean_[len_idx]), float(np.mean(raw_valid)), places=5)
        sentinel_rows = uad.X_train[:, len_idx] == -1.0
        self.assertTrue(np.all(X_tr[sentinel_rows, len_idx] == 0.0))

        # Binary availability indicators must remain exact 0/1
        for bcol in ("has_ip", "has_mqtt", "has_mqtt_topic", "has_mqtt_msg"):
            bidx = uad.feature_names.index(bcol)
            np.testing.assert_array_equal(X_tr[:, bidx], uad.X_train[:, bidx])

        # MQTTset binary indicator preserved
        mset = load_mqttset(self.set_dir)
        mscaler = MaskedStandardScaler(
            mset.feature_names, sentinel_value=None, preserve_cols=MQTTSET_PRESERVE_BINARY_COLS
        )
        X_mset = mscaler.fit_transform(mset.X_train)
        inf_idx = mset.feature_names.index("has_inf_rate")
        np.testing.assert_array_equal(X_mset[:, inf_idx], mset.X_train[:, inf_idx])

    def test_7_constant_and_all_missing_features(self) -> None:
        uad = load_mqtt_uad(self.uad_dir)
        scaler = MaskedStandardScaler(
            uad.feature_names, sentinel_value=-1.0, preserve_cols=UAD_PRESERVE_BINARY_COLS
        )
        X_tr = scaler.fit_transform(uad.X_train)
        X_te = scaler.transform(uad.X_test)

        miss_idx = uad.feature_names.index("mqtt.all_missing")
        const_idx = uad.feature_names.index("mqtt.const_val")
        self.assertTrue(np.all(X_tr[:, miss_idx] == 0.0))
        self.assertTrue(np.all(X_te[:, miss_idx] == 0.0))
        self.assertTrue(np.all(X_tr[:, const_idx] == 0.0))
        self.assertTrue(np.all(X_te[:, const_idx] == 0.0))
        self.assertFalse(np.any(np.isnan(X_tr)) or np.any(np.isinf(X_tr)))
        self.assertFalse(np.any(np.isnan(X_te)) or np.any(np.isinf(X_te)))

    def test_8_no_sequence_windows_cross_splits_or_files(self) -> None:
        uad = load_mqtt_uad(self.uad_dir)
        # Train has DoS.csv (4 rows -> 2 windows of W=3) and MitM.csv (3 rows -> 1 window of W=3)
        X_tr_seq, y_tr_seq, g_tr, r_tr = build_mqtt_uad_sequences(
            uad.X_train, uad.y_train, uad.train_groups, uad.train_row_ids, window_size=3
        )
        self.assertEqual(X_tr_seq.shape, (3, 3, 10))
        self.assertEqual(g_tr.tolist(), ["DoS.csv", "DoS.csv", "MitM.csv"])
        self.assertEqual(r_tr.tolist(), [2, 3, 2])

        # Test has DoS.csv (2 rows < W=3 -> 0 windows) and MitM.csv (3 rows -> 1 window)
        X_te_seq, _, g_te, r_te = build_mqtt_uad_sequences(
            uad.X_test, uad.y_test, uad.test_groups, uad.test_row_ids, window_size=3
        )
        self.assertEqual(X_te_seq.shape, (1, 3, 10))
        self.assertEqual(g_te.tolist(), ["MitM.csv"])
        self.assertEqual(r_te.tolist(), [7])

        # Short split where ALL groups < window_size must not crash
        X_short, y_short, _, _ = build_mqtt_uad_sequences(
            uad.X_test, uad.y_test, uad.test_groups, uad.test_row_ids, window_size=10
        )
        self.assertEqual(X_short.shape, (0, 10, 10))
        self.assertEqual(y_short.shape, (0,))

    def test_9_correct_sequence_labels(self) -> None:
        uad = load_mqtt_uad(self.uad_dir, binary=False)
        _, y_last, _, _ = build_mqtt_uad_sequences(
            uad.X_train, uad.y_train, uad.train_groups, uad.train_row_ids, window_size=3, label_mode="last"
        )
        # DoS.csv rows [0,1,2]->DoS(1), [1,2,3]->DoS(1); MitM.csv rows [0,1,2]->normal(0)
        self.assertEqual(y_last.tolist(), [1, 1, 0])

        uad_bin = load_mqtt_uad(self.uad_dir, binary=True)
        _, y_any, _, _ = build_mqtt_uad_sequences(
            uad_bin.X_train,
            uad_bin.y_train,
            uad_bin.train_groups,
            uad_bin.train_row_ids,
            window_size=3,
            label_mode="any_attack",
        )
        # MitM.csv [0,1,2] has mitm at 0,1 and normal at 2 -> any_attack = 1
        self.assertEqual(y_any.tolist(), [1, 1, 1])

    def test_10_reproducibility(self) -> None:
        uad1 = load_mqtt_uad(self.uad_dir)
        uad2 = load_mqtt_uad(self.uad_dir)
        np.testing.assert_array_equal(uad1.X_train, uad2.X_train)
        np.testing.assert_array_equal(uad1.y_train, uad2.y_train)
        w1 = compute_class_weights(uad1.y_train, len(MQTT_UAD_CLASSES))
        w2 = compute_class_weights(uad2.y_train, len(MQTT_UAD_CLASSES))
        np.testing.assert_array_equal(w1, w2)


if __name__ == "__main__":
    unittest.main()

