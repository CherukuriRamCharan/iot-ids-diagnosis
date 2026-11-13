"""Unit tests for Stage 3B.6 runtime optimizations (atomic checkpoints, resume, chunking, n_jobs)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from ids.datasets_real import load_mqtt_uad
from ids.train_baselines_improved import (
    ALL_FAMILIES,
    UAD_TASKS,
    build_checkpoint_meta,
    default_n_jobs,
    derive_binary_dataset,
    generate_improvement_report,
    load_valid_checkpoint,
    resolve_n_jobs,
    run_stage3b6,
    save_json_atomic,
)
from tests.test_stage3b_baselines import _write_synthetic_uad


def _write_synthetic_baseline_csv(path: Path) -> None:
    rows = []
    for task in UAD_TASKS:
        rows.append(
            {
                "task": task.task_id,
                "model": "RandomForest",
                "macro_f1_mean": 0.5,
                "balanced_accuracy_mean": 0.5,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


class TestStage3B6Runtime(unittest.TestCase):
    def test_n_jobs_resolution(self) -> None:
        d = default_n_jobs()
        expected = max(1, (os.cpu_count() or 4) - 2)
        self.assertEqual(d, expected)
        self.assertGreaterEqual(d, 1)
        self.assertEqual(resolve_n_jobs(None), d)
        self.assertEqual(resolve_n_jobs(-1), d)
        self.assertEqual(resolve_n_jobs(0), 1)
        self.assertEqual(resolve_n_jobs(2), 2)

    def test_derive_binary_dataset_matches_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            uad_dir = Path(tmp) / "uad"
            uad_dir.mkdir(parents=True, exist_ok=True)
            _write_synthetic_uad(uad_dir)

            ds_mc = load_mqtt_uad(uad_dir, binary=False, mqtt_only=False)
            ds_bin_direct = load_mqtt_uad(uad_dir, binary=True, mqtt_only=False)
            y_mc_train_before = ds_mc.y_train.copy()
            ds_bin = derive_binary_dataset(ds_mc)

            self.assertEqual(ds_bin.name, f"{ds_mc.name}_binary")
            self.assertEqual(ds_bin.class_names, ("normal", "attack"))
            self.assertEqual(ds_bin.feature_names, ds_mc.feature_names)
            np.testing.assert_array_equal(ds_bin.X_train, ds_mc.X_train)
            np.testing.assert_array_equal(ds_bin.X_val, ds_mc.X_val)
            np.testing.assert_array_equal(ds_bin.X_test, ds_mc.X_test)
            np.testing.assert_array_equal(ds_bin.y_train, ds_bin_direct.y_train)
            np.testing.assert_array_equal(ds_bin.y_val, ds_bin_direct.y_val)
            np.testing.assert_array_equal(ds_bin.y_test, ds_bin_direct.y_test)
            np.testing.assert_array_equal(ds_bin.y_train, (ds_mc.y_train > 0).astype(np.int64))
            np.testing.assert_array_equal(ds_bin.y_val, (ds_mc.y_val > 0).astype(np.int64))
            np.testing.assert_array_equal(ds_bin.y_test, (ds_mc.y_test > 0).astype(np.int64))
            np.testing.assert_array_equal(ds_mc.y_train, y_mc_train_before)
            np.testing.assert_array_equal(ds_bin.train_groups, ds_mc.train_groups)
            np.testing.assert_array_equal(ds_bin.val_groups, ds_mc.val_groups)
            np.testing.assert_array_equal(ds_bin.test_groups, ds_mc.test_groups)
            np.testing.assert_array_equal(ds_bin.train_row_ids, ds_mc.train_row_ids)
            np.testing.assert_array_equal(ds_bin.val_row_ids, ds_mc.val_row_ids)
            np.testing.assert_array_equal(ds_bin.test_row_ids, ds_mc.test_row_ids)
            self.assertEqual(ds_bin.metadata, ds_mc.metadata)
            # Verify composite (source_file, source_row_idx) identities are unique within
            # each split, identical to both ds_mc and ds_bin_direct, and disjoint across splits.
            train_ids = [(str(g), int(r)) for g, r in zip(ds_bin.train_groups, ds_bin.train_row_ids)]
            val_ids = [(str(g), int(r)) for g, r in zip(ds_bin.val_groups, ds_bin.val_row_ids)]
            test_ids = [(str(g), int(r)) for g, r in zip(ds_bin.test_groups, ds_bin.test_row_ids)]
            mc_train_ids = [(str(g), int(r)) for g, r in zip(ds_mc.train_groups, ds_mc.train_row_ids)]
            mc_val_ids = [(str(g), int(r)) for g, r in zip(ds_mc.val_groups, ds_mc.val_row_ids)]
            mc_test_ids = [(str(g), int(r)) for g, r in zip(ds_mc.test_groups, ds_mc.test_row_ids)]
            dir_train_ids = [(str(g), int(r)) for g, r in zip(ds_bin_direct.train_groups, ds_bin_direct.train_row_ids)]
            dir_val_ids = [(str(g), int(r)) for g, r in zip(ds_bin_direct.val_groups, ds_bin_direct.val_row_ids)]
            dir_test_ids = [(str(g), int(r)) for g, r in zip(ds_bin_direct.test_groups, ds_bin_direct.test_row_ids)]

            self.assertEqual(train_ids, mc_train_ids)
            self.assertEqual(val_ids, mc_val_ids)
            self.assertEqual(test_ids, mc_test_ids)
            self.assertEqual(train_ids, dir_train_ids)
            self.assertEqual(val_ids, dir_val_ids)
            self.assertEqual(test_ids, dir_test_ids)

            self.assertEqual(len(train_ids), len(set(train_ids)))
            self.assertEqual(len(val_ids), len(set(val_ids)))
            self.assertEqual(len(test_ids), len(set(test_ids)))

            self.assertTrue(set(train_ids).isdisjoint(set(val_ids)))
            self.assertTrue(set(train_ids).isdisjoint(set(test_ids)))
            self.assertTrue(set(val_ids).isdisjoint(set(test_ids)))

    def test_save_json_atomic_and_checkpoint_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            uad_dir = tmp_dir / "uad"
            uad_dir.mkdir(parents=True, exist_ok=True)
            _write_synthetic_uad(uad_dir)
            ds = load_mqtt_uad(uad_dir, binary=False, mqtt_only=False)

            ckpt_path = tmp_dir / "checkpoints" / "sample.json"
            meta = build_checkpoint_meta(
                ds,
                task_id="mqtt_uad_full_multiclass",
                model_family="RF_tuned",
                seed=42,
                fast_mode=True,
            )
            payload = {
                "meta": meta,
                "metrics": {
                    "selected_config": {"n_estimators": 20},
                    "macro_f1": 0.75,
                },
            }
            save_json_atomic(ckpt_path, payload)
            self.assertTrue(ckpt_path.exists())
            self.assertEqual(list(ckpt_path.parent.glob("*.tmp")), [])

            loaded = load_valid_checkpoint(ckpt_path, meta)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["metrics"]["macro_f1"], 0.75)

            # Incompatible seed
            bad_meta_seed = dict(meta, seed=43)
            self.assertIsNone(load_valid_checkpoint(ckpt_path, bad_meta_seed))

            # Incompatible feature_names
            bad_meta_feat = dict(meta, feature_names=meta["feature_names"][:-1])
            self.assertIsNone(load_valid_checkpoint(ckpt_path, bad_meta_feat))

            # Incompatible split_sizes
            bad_meta_split = dict(meta, split_sizes={"train": 999, "val": 1, "test": 1})
            self.assertIsNone(load_valid_checkpoint(ckpt_path, bad_meta_split))

            # Missing required metrics field
            save_json_atomic(ckpt_path, {"meta": meta, "metrics": {"macro_f1": 0.5}})
            self.assertIsNone(load_valid_checkpoint(ckpt_path, meta))

            # Corrupted JSON
            ckpt_path.write_text("{not valid json", encoding="utf-8")
            self.assertIsNone(load_valid_checkpoint(ckpt_path, meta))

    def test_resume_skips_completed_checkpoints_and_chunked_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uad_dir = root / "uad"
            out_dir = root / "out"
            report_path = root / "REPORT.md"
            baseline_csv = root / "baseline.csv"
            uad_dir.mkdir(parents=True, exist_ok=True)
            _write_synthetic_uad(uad_dir)
            _write_synthetic_baseline_csv(baseline_csv)

            expected_tasks = (UAD_TASKS[0], UAD_TASKS[1])
            expected_families = ("RF_tuned", "HistGB_tuned")
            seeds = (42,)

            # Chunk 1: run only task 0, while expecting tasks 0 and 1
            summary_partial = run_stage3b6(
                uad_dir=uad_dir,
                out_dir=out_dir,
                report_path=report_path,
                baseline_csv=baseline_csv,
                seeds=seeds,
                tasks=(UAD_TASKS[0],),
                families=expected_families,
                n_jobs=1,
                fast_mode=True,
                resume=False,
                expected_tasks=expected_tasks,
                expected_families=expected_families,
            )
            self.assertTrue(summary_partial["incomplete_experiments"])
            report_text_partial = report_path.read_text(encoding="utf-8")
            self.assertIn("WARNING — INCOMPLETE EXPERIMENT RUN:", report_text_partial)

            ckpt_file = out_dir / "checkpoints" / f"{UAD_TASKS[0].task_id}__RF_tuned__seed42.json"
            self.assertTrue(ckpt_file.exists())

            # Chunk 2: run task 1 with resume=True
            summary_chunk2 = run_stage3b6(
                uad_dir=uad_dir,
                out_dir=out_dir,
                report_path=report_path,
                baseline_csv=baseline_csv,
                seeds=seeds,
                tasks=(UAD_TASKS[1],),
                families=expected_families,
                n_jobs=1,
                fast_mode=True,
                resume=True,
                expected_tasks=expected_tasks,
                expected_families=expected_families,
            )
            self.assertEqual(summary_chunk2["incomplete_experiments"], [])

            # Re-run task 0 with resume=True and verify fit_select_and_eval is not called
            with mock.patch("ids.train_baselines_improved.fit_select_and_eval") as mocked_fit:
                summary_resumed = run_stage3b6(
                    uad_dir=uad_dir,
                    out_dir=out_dir,
                    report_path=report_path,
                    baseline_csv=baseline_csv,
                    seeds=seeds,
                    tasks=(UAD_TASKS[0],),
                    families=expected_families,
                    n_jobs=1,
                    fast_mode=True,
                    resume=True,
                    expected_tasks=expected_tasks,
                    expected_families=expected_families,
                )
                mocked_fit.assert_not_called()
                self.assertEqual(summary_resumed["incomplete_experiments"], [])

            # Aggregate-only mode produces complete report without warning
            summary_agg = run_stage3b6(
                uad_dir=uad_dir,
                out_dir=out_dir,
                report_path=report_path,
                baseline_csv=baseline_csv,
                seeds=seeds,
                tasks=expected_tasks,
                families=expected_families,
                n_jobs=1,
                fast_mode=True,
                aggregate_only=True,
                expected_tasks=expected_tasks,
                expected_families=expected_families,
            )
            self.assertEqual(summary_agg["incomplete_experiments"], [])
            report_text_final = report_path.read_text(encoding="utf-8")
            self.assertNotIn("WARNING — INCOMPLETE EXPERIMENT RUN:", report_text_final)
            self.assertTrue((out_dir / "aggregated_metrics.json").exists())
            self.assertTrue((out_dir / "threshold_tuning.json").exists())

    def test_generate_improvement_report_missing_dataset_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Case 1: baseline CSV has 'task' (derivable) but no 'dataset_group'
            baseline_csv = root / "baseline_derivable.csv"
            _write_synthetic_baseline_csv(baseline_csv)
            report_path = root / "REPORT1.md"
            generate_improvement_report({}, [], {}, baseline_csv, (42,), report_path)
            text1 = report_path.read_text(encoding="utf-8")
            self.assertIn("[Stage 3B] RandomForest", text1)
            self.assertIn("N/A", text1)

            # Case 2: baseline CSV has no 'dataset_group', 'task_id', 'task', or 'dataset_label'
            unknown_csv = root / "baseline_unknown.csv"
            pd.DataFrame([{"model": "RandomForest", "macro_f1_mean": 0.5}]).to_csv(unknown_csv, index=False)
            report_path2 = root / "REPORT2.md"
            generate_improvement_report({}, [], {}, unknown_csv, (42,), report_path2)
            text2 = report_path2.read_text(encoding="utf-8")
            self.assertIn("dataset_group", text2)
            self.assertIn("N/A", text2)


if __name__ == "__main__":
    unittest.main()

