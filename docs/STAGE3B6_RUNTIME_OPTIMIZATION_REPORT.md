# Stage 3B.6 — Runtime Optimization Report

## 1. Files Changed and Created

- **Modified**: `ids/train_baselines_improved.py`
- **Created**: `tests/test_stage3b6_runtime.py`
- **Created**: `docs/STAGE3B6_RUNTIME_OPTIMIZATION_REPORT.md`

No datasets (`datasets/raw/`, `datasets/processed/`), Stage 3A loaders (`ids/datasets_real.py`), immutable Stage 3B baseline results (`results/stage_3b_baselines/`), existing test files (`tests/test_stage3b6_improvements.py`), or FSM/simulation/diagnosis modules were modified.

---

## 2. Implementation Changes in `ids/train_baselines_improved.py`

1. **Atomic Checkpointing and Validation**
   - Added `save_json_atomic(path, payload)` which writes to a temporary `.tmp` file in the target directory and atomically replaces the target path via `os.replace()`.
   - Added `build_checkpoint_meta(ds, task_id, model_family, seed, fast_mode)` recording `task_id`, `model_family`, `seed`, `fast_mode`, `dataset_name`, `feature_names`, `class_names`, and `split_sizes`.
   - Added `load_valid_checkpoint(path, expected_meta)` which rejects corrupted JSON, checkpoints missing required `metrics.selected_config` or `metrics.macro_f1` fields, or checkpoints with mismatched metadata (printing `[WARN] Rejecting incompatible/corrupted checkpoint ...` and returning `None`).

2. **Bounded CPU Parallelism (`--n-jobs`)**
   - Added `default_n_jobs()` returning `max(1, (os.cpu_count() or 4) - 2)` and `resolve_n_jobs(n_jobs)` (passing `None` or `< 0` resolves to `default_n_jobs()`, `0` resolves to `1`).
   - Added `--n-jobs` CLI flag to let the user override CPU worker count.
   - `HistGradientBoostingClassifier` runs sequentially inside a single process without nested OpenMP/joblib oversubscription, and tree ensembles (`RandomForestClassifier`, `ExtraTreesClassifier`) use the resolved bounded `n_jobs` per fit without nested grid parallelism.

3. **Per-Fit Progress Reporting**
   - Updated `_run_grid(...)` inside `fit_select_and_eval` to print flushed progress lines per hyperparameter configuration:
     `[{tag}] fit {idx}/{total} cfg={cfg} -> val_macro_f1={val_f1:.4f} ({dt:.2f}s)`

4. **Single-Load Dataset Caching and In-Memory Binary Derivation**
   - Added `derive_binary_dataset(ds_mc)` to construct the binary `SplitDataset` (`normal` = 0, `attack` = 1) directly in memory from the loaded multiclass `SplitDataset`, preserving `X_*`, `feature_names`, `train_groups`/`val_groups`/`test_groups`, and `train_row_ids`/`val_row_ids`/`test_row_ids`.
   - Added `_get_task_dataset` inside `run_stage3b6` so each `mqtt_only` setting (`False` and `True`) loads the CSV once rather than reloading for each task/seed.

5. **Chunked Execution, Resume, and Aggregation**
   - Added CLI flags `--tasks`, `--families`, `--resume`, and `--aggregate-only`.
   - When `--resume` is set, valid existing checkpoints are loaded and skipped without re-fitting.
   - When `--aggregate-only` is set, existing valid checkpoints in `checkpoints/` are aggregated into `aggregated_metrics.json`, `threshold_tuning.json`, `per_seed_metrics.csv`, `aggregated_metrics.csv`, and `docs/STAGE3B6_BASELINE_IMPROVEMENT_REPORT.md` without fitting any models.

6. **Incomplete Experiment Detection**
   - `run_stage3b6` checks the full expected `(expected_tasks × expected_families × seeds)` matrix against loaded/completed results and records any missing seeds in `incomplete_experiments`.
   - When `incomplete_experiments` is non-empty, `generate_improvement_report` emits a prominent `> **WARNING — INCOMPLETE EXPERIMENT RUN:**` callout at the top of `docs/STAGE3B6_BASELINE_IMPROVEMENT_REPORT.md` listing the missing `(task, model)` pairs and missing seeds.

---

## 3. Checkpoint Directory and Naming Convention

All checkpoints are stored under:
- `results/stage_3b6_improvements/checkpoints/`

Naming convention:
- Model-task-seed checkpoints:
  `{task_id}__{model_family}__seed{seed}.json`
  (e.g., `mqtt_uad_full_multiclass__RF_tuned__seed42.json`)
- Feature ablation checkpoint:
  `feature_ablation__seed{seeds[0]}.json`

Incomplete fits never write a checkpoint; only after all hyperparameter evaluations and final test evaluation finish is the checkpoint written atomically via `save_json_atomic`.

---

## 4. Unit Tests Added (`tests/test_stage3b6_runtime.py`)

`tests/test_stage3b6_runtime.py` tests:
- `default_n_jobs()` and `resolve_n_jobs()` bounds (`None`, `-1`, `0`, positive integers)
- `derive_binary_dataset()` equivalence against `load_mqtt_uad(..., task="binary")`
- `save_json_atomic()` (no leftover `.tmp` files) and `load_valid_checkpoint()` rejection of corrupted JSON, missing metric fields, and mismatched `seed`, `feature_names`, or `split_sizes`
- Chunked `run_stage3b6()` with `expected_tasks` detecting `incomplete_experiments` and writing the report warning, `--resume` skipping already-checkpointed runs (verified by mocking `fit_select_and_eval`), and `--aggregate-only` producing a complete report once all chunks finish.

> **Execution Note:** Because the assistant's shell runner cannot execute commands in this Windows environment, these unit tests have **not** been executed yet. Run them manually in `cmd.exe` from the project root:
> ```cmd
> python -m unittest tests/test_stage3b6_runtime.py -v
> python -m unittest tests/test_stage3b6_improvements.py -v
> ```

---

## 5. Exact Commands for Chunked Execution, Resume, and Aggregation

Run all commands from `C:\Users\cheru\Downloads\iot-ids-diagnosis\iot-ids-diagnosis` in `cmd.exe`.

### Option A: Chunked Execution by Task (Recommended for Long Runs)
```cmd
python -m ids.train_baselines_improved --tasks mqtt_uad_full_multiclass --resume
python -m ids.train_baselines_improved --tasks mqtt_uad_full_binary --resume
python -m ids.train_baselines_improved --tasks mqtt_uad_mqtt_only_multiclass --resume
python -m ids.train_baselines_improved --tasks mqtt_uad_mqtt_only_binary --resume
```

### Option B: Chunked Execution by Task and Model Family
For example, running `RF_tuned` and `ExtraTrees_tuned` separately on `mqtt_uad_full_multiclass`:
```cmd
python -m ids.train_baselines_improved --tasks mqtt_uad_full_multiclass --families RF_tuned --resume
python -m ids.train_baselines_improved --tasks mqtt_uad_full_multiclass --families ExtraTrees_tuned --resume
python -m ids.train_baselines_improved --tasks mqtt_uad_full_multiclass --families HistGB_tuned RF_train_oversampled --resume
```

### Final Aggregation Command
Once all tasks, families, and seeds (`42 43 44 45 46`) have completed checkpoints:
```cmd
python -m ids.train_baselines_improved --aggregate-only
```

### Resuming After Interruption (`Ctrl+C`)
If any run is interrupted mid-way:
1. Any `.tmp` file from an interrupted atomic write is ignored; all previously completed `(task_id, model_family, seed)` JSON files in `results/stage_3b6_improvements/checkpoints/` remain intact.
2. Re-run the exact same command with `--resume`. Valid checkpoints are loaded and skipped immediately, while the interrupted `(task_id, model_family, seed)` is re-run cleanly.

### How Incomplete Experiments Are Detected
- During aggregation (`run_stage3b6` or `--aggregate-only`), the runner checks every expected `(task_id, model_family, seed)` across all 4 `UAD_TASKS`, all 4 `ALL_FAMILIES`, and all requested `seeds`.
- Any missing `(task_id, model_family)` or partial seed list is recorded in `incomplete_experiments` and triggers a `WARNING — INCOMPLETE EXPERIMENT RUN:` banner in `docs/STAGE3B6_BASELINE_IMPROVEMENT_REPORT.md` so partial results are never mistaken for a complete 5-seed run.

