"""Unit tests for Stage 3B classical ML baselines (ids/train_baselines.py).

Uses small synthetic datasets in temporary directories; never trains on the full
raw/processed datasets during unit tests.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ids.datasets_real import load_mqtt_uad
from ids.train_baselines import (
    TaskSpec,
    compute_classification_metrics,
    run_stage3b,
    train_and_eval_decision_tree,
    train_and_eval_random_forest,
)


def _write_synthetic_uad(root: Path) -> None:
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
        "type",
        "split",
    ]
    rows = []
    # Train: imbalanced (10 normal, 6 DoS, 2 mitm, 2 intrusion)
    for i in range(10):
        rows.append(("DoS.csv", i, 0.5, 60.0, 1, 1 if i % 2 == 0 else 0, 1, 1, 3.0 if i % 2 == 0 else -1.0, 12.0 if i % 2 == 0 else -1.0, "normal", "train"))
    for i in range(10, 16):
        rows.append(("DoS.csv", i, 0.01, 120.0, 1, 1, 1, 1, 3.0, 48.0, "DoS", "train"))
    for i in range(2):
        rows.append(("MitM.csv", i, 0.2, 42.0, 0, 0, 0, 0, -1.0, -1.0, "mitm", "train"))
    for i in range(2):
        rows.append(("Intrusion.csv", i, 0.3, 90.0, 1, 1, 1, 1, 1.0, 30.0, "intrusion", "train"))

    # Val: all 4 classes
    rows.extend([
        ("DoS.csv", 16, 0.4, 62.0, 1, 1, 1, 1, 3.0, 10.0, "normal", "val"),
        ("DoS.csv", 17, 0.02, 118.0, 1, 1, 1, 1, 3.0, 50.0, "DoS", "val"),
        ("MitM.csv", 2, 0.25, 42.0, 0, 0, 0, 0, -1.0, -1.0, "mitm", "val"),
        ("Intrusion.csv", 2, 0.35, 88.0, 1, 1, 1, 1, 1.0, 28.0, "intrusion", "val"),
    ])

    # Test: deliberately omit 'mitm' in MQTT-only test (mitm has has_mqtt=0)
    rows.extend([
        ("DoS.csv", 18, 0.45, 64.0, 1, 1, 1, 1, 3.0, 14.0, "normal", "test"),
        ("DoS.csv", 19, 0.01, 122.0, 1, 1, 1, 1, 3.0, 52.0, "DoS", "test"),
        ("MitM.csv", 3, 0.22, 42.0, 0, 0, 0, 0, -1.0, -1.0, "mitm", "test"),
        ("Intrusion.csv", 3, 0.32, 92.0, 1, 1, 1, 1, 1.0, 32.0, "intrusion", "test"),
    ])

    pd.DataFrame(rows, columns=cols).to_csv(root / "mqtt_uad_features.csv", index=False)
    manifest = {
        "feature_columns": [
            "frame.time_delta",
            "frame.len",
            "has_ip",
            "has_mqtt",
            "has_mqtt_topic",
            "has_mqtt_msg",
            "mqtt.msgtype",
            "mqtt.len",
        ]
    }
    (root / "mqtt_uad_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_synthetic_mqttset(root: Path) -> None:
    cols = ["source_row_idx", "Flow_Duration", "Tot_Fwd_Pkts", "has_inf_rate", "label", "Cat", "split"]
    rows = [
        (0, 100.0, 2.0, 0, "Normal", "Normal", "train"),
        (1, 110.0, 2.0, 0, "Normal", "Normal", "train"),
        (2, 10.0, 50.0, 0, "Anomaly", "Flood", "train"),
        (3, 12.0, 48.0, 0, "Anomaly", "MalariaDoS", "train"),
        (4, 50.0, 8.0, 0, "Anomaly", "Bruteforce", "train"),
        (5, 80.0, 3.0, 1, "Anomaly", "Malformed", "train"),
        (6, 500.0, 1.0, 0, "Anomaly", "Slowite", "train"),
        (7, 105.0, 2.0, 0, "Normal", "Normal", "val"),
        (8, 11.0, 49.0, 0, "Anomaly", "Flood", "val"),
        (9, 102.0, 2.0, 0, "Normal", "Normal", "test"),
        (10, 13.0, 47.0, 0, "Anomaly", "Flood", "test"),
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(root / "mqttset_features.csv", index=False)
    pd.DataFrame({"Flow_ID": [f"f{i}" for i in range(len(df))]}).to_csv(
        root / "mqttset_metadata.csv", index=False
    )
    manifest = {"feature_columns": ["Flow_Duration", "Tot_Fwd_Pkts", "has_inf_rate"]}
    (root / "mqttset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class TestStage3BBaselines(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.uad_dir = self.tmp / "mqtt_uad"
        self.set_dir = self.tmp / "mqttset"
        self.out_dir = self.tmp / "out"
        self.uad_dir.mkdir()
        self.set_dir.mkdir()
        _write_synthetic_uad(self.uad_dir)
        _write_synthetic_mqttset(self.set_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_rf_and_dt_reproducibility_and_metrics(self) -> None:
        ds = load_mqtt_uad(self.uad_dir, binary=False, mqtt_only=False)
        rf1 = train_and_eval_random_forest(ds, seed=42, n_estimators=10, max_depth=5, n_jobs=1)
        rf2 = train_and_eval_random_forest(ds, seed=42, n_estimators=10, max_depth=5, n_jobs=1)
        self.assertEqual(rf1["macro_f1"], rf2["macro_f1"])
        self.assertEqual(rf1["confusion_matrix"], rf2["confusion_matrix"])
        self.assertGreaterEqual(rf1["train_time_s"], 0.0)
        self.assertGreaterEqual(rf1["inference_time_s"], 0.0)

        dt1 = train_and_eval_decision_tree(ds, seed=42, depth_grid=(4, 8))
        dt2 = train_and_eval_decision_tree(ds, seed=42, depth_grid=(4, 8))
        self.assertEqual(dt1["macro_f1"], dt2["macro_f1"])
        self.assertEqual(len(dt1["confusion_matrix"]), 4)

    def test_missing_class_and_leakage_guard(self) -> None:
        # mitm has has_mqtt=0, so it is absent in mqtt_only test split
        ds_mqtt = load_mqtt_uad(self.uad_dir, binary=False, mqtt_only=True)
        rf = train_and_eval_random_forest(ds_mqtt, seed=42, n_estimators=10, max_depth=5, n_jobs=1)
        self.assertIn("mitm", rf["absent_test_classes"])
        self.assertEqual(len(rf["confusion_matrix"]), 4)
        self.assertEqual(rf["per_class"]["mitm"]["support"], 0)

        # Direct metric check for absent class
        m = compute_classification_metrics(
            np.array([0, 1]), np.array([0, 1]), ("normal", "DoS", "mitm", "intrusion")
        )
        self.assertEqual(m["absent_test_classes"], ["mitm", "intrusion"])

        # Leakage column injection must fail
        ds_bad = load_mqtt_uad(self.uad_dir, binary=True)
        ds_bad.feature_names = ("frame.len", "ip.src")
        with self.assertRaises(ValueError):
            train_and_eval_random_forest(ds_bad, seed=42, n_estimators=5, n_jobs=1)

    def test_end_to_end_stage3b_outputs(self) -> None:
        tasks = (
            TaskSpec("mqtt_uad_full_multiclass", "MQTT-UAD", "MQTT-UAD Full 4-Class", binary=False, mqtt_only=False),
            TaskSpec("mqtt_uad_mqtt_only_binary", "MQTT-UAD", "MQTT-UAD MQTT-Only Binary", binary=True, mqtt_only=True),
            TaskSpec("mqttset_multiclass", "MQTTset", "MQTTset 6-Class", binary=False),
        )
        res = run_stage3b(
            uad_dir=self.uad_dir,
            mqttset_dir=self.set_dir,
            out_dir=self.out_dir,
            seeds=(42, 43),
            tasks=tasks,
            rf_estimators=10,
            rf_max_depth=6,
            rf_n_jobs=1,
            dt_depth_grid=(4, 8),
        )
        self.assertEqual(len(res["aggregated"]), 6)
        self.assertTrue((self.out_dir / "per_seed_metrics.json").exists())
        self.assertTrue((self.out_dir / "aggregated_metrics.json").exists())
        self.assertTrue((self.out_dir / "model_comparison.csv").exists())
        self.assertTrue((self.out_dir / "STAGE_3B_BASELINE_REPORT.md").exists())
        self.assertTrue((self.out_dir / "confusion_matrices" / "mqtt_uad_full_multiclass__RandomForest.png").exists())
        self.assertTrue((self.out_dir / "confusion_matrices" / "mqttset_multiclass__DecisionTree.json").exists())


if __name__ == "__main__":
    unittest.main()

