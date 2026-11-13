# Stage 3B.6 — Runtime Optimization Unit Test Fix Report

## 1. Root Causes

### Round 1 Fixes (Already Applied)
1. **Stage 3A Loader API Mismatch**: `tests/test_stage3b6_runtime.py` called `load_mqtt_uad(..., task=...)`, whereas the actual Stage 3A signature in `ids/datasets_real.py` uses `binary: bool = False`.
2. **Missing `dataset_group` in Baseline CSV**: `generate_improvement_report()` in `ids/train_baselines_improved.py` assumed `dataset_group` and several optional columns always existed in `baseline_csv`, raising `KeyError: 'dataset_group'` on minimal synthetic CSVs.

### Round 2 Fixes (Remaining Three Failures in `tests.test_stage3b6_runtime`)
1. **Failure 1 (`test_derive_binary_dataset_matches_loader`: `AssertionError: 'mqtt_uad_binary' != 'mqtt_uad'`)**
   - `load_mqtt_uad()` sets `name="mqtt_uad"` (or `"mqtt_uad_mqtt_only"`), while `derive_binary_dataset(ds_mc)` intentionally sets `name=f"{ds_mc.name}_binary"` (`"mqtt_uad_binary"`). The test erroneously asserted `ds_bin.name == ds_mc.name`.
2. **Failure 2 (`test_generate_improvement_report_missing_dataset_group`: `AssertionError: '[Stage 3B] RandomForest' not found`)**
   - `_write_synthetic_baseline_csv()` in `tests/test_stage3b6_runtime.py` stored `"task": task` where `task` is a `UADTask` dataclass instance rather than `task.task_id`. Consequently, `pd.DataFrame.to_csv()` serialized `"UADTask(task_id='mqtt_uad_full_multiclass', ...)"`, which did not start with `"mqtt_uad"`, so `generate_improvement_report()` legitimately excluded those rows when deriving `MQTT-UAD` tasks.
3. **Failure 3 (`test_resume_skips_completed_checkpoints_and_chunked_aggregation`: `AssertionError: False is not true` at `ckpt_file.exists()`)**
   - The test constructed `ckpt_file = out_dir / "checkpoints" / f"{UAD_TASKS[0]}__RF_tuned__seed42.json"`, interpolating the `UADTask` dataclass object instead of `UAD_TASKS[0].task_id`. Meanwhile, `run_stage3b6()` wrote the actual checkpoint to `out_dir / "checkpoints" / "mqtt_uad_full_multiclass__RF_tuned__seed42.json"`.

### Round 3 Fix (Row ID Split Integrity)
- **Failure (`test_derive_binary_dataset_matches_loader`: `set(ds_bin.train_row_ids).isdisjoint(set(ds_bin.val_row_ids))`)**:
  - **Investigation**: In `ids/datasets_real.py` (`load_mqtt_uad()`), `train_groups` / `val_groups` / `test_groups` store `source_file` and `train_row_ids` / `val_row_ids` / `test_row_ids` store `source_row_idx`. In MQTT-UAD (and `_write_synthetic_uad` in `tests/test_stage3b_baselines.py`), `source_row_idx` is 0-indexed **within each source file** (`DoS.csv`, `MitM.csv`, `Intrusion.csv`), not globally unique across files (e.g., `DoS.csv` row `2` is in `train`, while `MitM.csv` row `2` and `Intrusion.csv` row `2` are in `val`).
  - **Root Cause**: The test checked `set(row_ids)` alone instead of the composite sample identity `(source_file, source_row_idx)` = `(group, row_id)`. Neither `load_mqtt_uad()` nor `derive_binary_dataset()` moved any sample across splits.
  - **Fix**: Updated `test_derive_binary_dataset_matches_loader` in `tests/test_stage3b6_runtime.py` to construct composite `(str(group), int(row_id))` identities for each split, assert exact ordered identity equality across `ds_bin`, `ds_mc`, and `ds_bin_direct`, assert zero duplicate composite identities within each split, and assert strict pairwise disjointness across `(train, val)`, `(train, test)`, and `(val, test)`. Stage 3A loaders were not modified.

---

## 2. Exact Files Changed

- **Modified**: `tests/test_stage3b6_runtime.py`
  - `_write_synthetic_baseline_csv()`: changed `"task": task` to `"task": task.task_id`.
  - `test_derive_binary_dataset_matches_loader()`: changed `self.assertEqual(ds_bin.name, ds_mc.name)` to `self.assertEqual(ds_bin.name, f"{ds_mc.name}_binary")`.
  - `test_resume_skips_completed_checkpoints_and_chunked_aggregation()`: changed `f"{UAD_TASKS[0]}__RF_tuned__seed42.json"` to `f"{UAD_TASKS[0].task_id}__RF_tuned__seed42.json"`.
- **Previously Modified (Round 1)**: `ids/train_baselines_improved.py` (`generate_improvement_report()` safe column inspection and `dataset_group` derivation without fabricating metrics).
- **Updated**: `docs/STAGE3B6_RUNTIME_TEST_FIX_REPORT.md`

---

## 3. Actual Checkpoint Path and Naming Convention

Verified in `ids/train_baselines_improved.py` (`run_stage3b6`):
- **Directory**: `{out_dir}/checkpoints/` (default: `results/stage_3b6_improvements/checkpoints/`)
- **Per-run model checkpoints**: `{task.task_id}__{fam}__seed{seed}.json`
  - Example: `results/stage_3b6_improvements/checkpoints/mqtt_uad_full_multiclass__RF_tuned__seed42.json`
- **Feature ablation checkpoint**: `feature_ablation__seed{seeds[0]}.json`
- **Atomic write**: written to `{path}.tmp` and atomically moved via `os.replace()`.

---

## 4. Test Execution Status

> **Note:** The assistant's shell runner cannot execute commands on this Windows machine (`powershell` executable not found in `%PATH%`), so the updated tests have **not** been executed by the assistant. Please run both commands in `cmd.exe` from `C:\Users\cheru\Downloads\iot-ids-diagnosis\iot-ids-diagnosis`:
>
> ```cmd
> python -m unittest tests.test_stage3b6_runtime -v
> python -m unittest tests.test_stage3b6_improvements -v
> ```

---

## 5. Confirmation

- **No full Stage 3B.6 experiment was run.**
- **Stage 3A loaders (`ids/datasets_real.py`), raw/processed datasets, original Stage 3B results (`results/stage_3b_baselines/`), and `tests/test_stage3b6_improvements.py` were not modified.**
- **All synthetic test outputs remain isolated inside `tempfile.TemporaryDirectory()`.**
