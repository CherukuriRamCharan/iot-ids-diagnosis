"""Unit tests for Stage 3B.6 controlled baseline improvements (ids/train_baselines_improved.py)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ids.train_baselines_improved import (
    UADTask,
    cap_oversample_train,
    run_stage3b6,
    tune_binary_threshold,
)
from tests.test_stage3b_baselines import _write_synthetic_uad


class TestStage3B6Improvements(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.uad_dir = self.tmp / "mqtt_uad"
        self.out_dir = self.tmp / "out"
        self.report_path = self.tmp / "STAGE3B6_REPORT.md"
        self.uad_dir.mkdir()
        _write_synthetic_uad(self.uad_dir)

        # Ensure all 33-feature ablation columns exist or subset works cleanly
        df = pd.read_csv(self.uad_dir / "mqtt_uad_features.csv")
        for extra in ("has_tcp", "is_mqtt_port", "frame.time_delta_displayed", "frame.cap_len", "mqtt.msgid", "mqtt.sub.qos", "mqtt.suback.qos"):
            df[extra] = 0.0
        df.to_csv(self.uad_dir / "mqtt_uad_features.csv", index=False)
        manifest = json.loads((self.uad_dir / "mqtt_uad_manifest.json").read_text(encoding="utf-8"))
        manifest["feature_columns"].extend(
            ["has_tcp", "is_mqtt_port", "frame.time_delta_displayed", "frame.cap_len", "mqtt.msgid", "mqtt.sub.qos", "mqtt.suback.qos"]
        )
        (self.uad_dir / "mqtt_uad_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_cap_oversample_train_only(self) -> None:
        X = np.arange(20, dtype=np.float32).reshape(10, 2)
        y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 2, 2])
        X_os, y_os = cap_oversample_train(X, y, target_cap=4, seed=42)
        counts = dict(zip(*np.unique(y_os, return_counts=True)))
        self.assertEqual(counts[0], 6)
        self.assertEqual(counts[1], 4)
        self.assertEqual(counts[2], 4)

    def test_tune_binary_threshold_on_val(self) -> None:
        y_val = np.array([0, 0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.35, 0.4, 0.8])
        tau, f1 = tune_binary_threshold(y_val, proba)
        self.assertGreaterEqual(tau, 0.35)
        self.assertLessEqual(tau, 0.40)
        self.assertAlmostEqual(f1, 1.0)

    def test_end_to_end_stage3b6_fast(self) -> None:
        tasks = (
            UADTask("mqtt_uad_full_multiclass", "MQTT-UAD Full 4-Class", binary=False, mqtt_only=False),
            UADTask("mqtt_uad_full_binary", "MQTT-UAD Full Binary", binary=True, mqtt_only=False),
        )
        res = run_stage3b6(
            uad_dir=self.uad_dir,
            out_dir=self.out_dir,
            report_path=self.report_path,
            baseline_csv=self.tmp / "nonexistent.csv",
            seeds=(42, 43),
            tasks=tasks,
            families=("RF_tuned", "HistGB_tuned", "RF_train_oversampled"),
            n_jobs=1,
            fast_mode=True,
        )
        self.assertEqual(len(res["aggregated"]), 6)
        self.assertEqual(len(res["ablation"]), 5)
        self.assertTrue((self.out_dir / "experiment_config.json").exists())
        self.assertTrue((self.out_dir / "per_seed_metrics.json").exists())
        self.assertTrue((self.out_dir / "aggregated_metrics.json").exists())
        self.assertTrue((self.out_dir / "feature_ablation.csv").exists())
        self.assertTrue((self.out_dir / "threshold_tuning.json").exists())
        self.assertTrue((self.out_dir / "model_comparison.csv").exists())
        self.assertTrue(self.report_path.exists())


if __name__ == "__main__":
    unittest.main()

