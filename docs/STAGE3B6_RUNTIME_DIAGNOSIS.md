# Stage 3B.6 — Runtime Diagnosis and Safe Optimization Plan

> **Incident:** `KeyboardInterrupt` during Stage 3B.6 execution at `ids/train_baselines_improved.py`, line 159 (`clf.fit(ds.X_train, ds.y_train)` inside `RF_tuned`).  
> **Scope:** Read-only performance diagnosis. No code or datasets were modified.

---

## 1. Root Cause & Exact Model-Fit Count

`run_stage3b6()` executes a full grid search **independently for every `(task, seed)` pair** across 4 tasks and 5 seeds (`20` pairs total):

| Model Family | Grid Size per `(task, seed)` | Fits across 4 Tasks × 5 Seeds | Estimator Cost per Fit |
|---|---:|---:|---|
| `RF_tuned` | `3 class_weight × 3 max_depth × 2 min_samples_leaf` = **18** | **360** | `150` trees (`54,000` trees total) |
| `ExtraTrees_tuned` | `3 class_weight × 2 max_depths × 2 min_samples_leaf` = **12** | **240** | `150` trees (`36,000` trees total) |
| `HistGB_tuned` | `2 class_weight × 3 max_depths × 2 learning_rates` = **12** | **240** | `150` boosting iters (up to `600` trees in 4-class) |
| `RF_train_oversampled` | `3 max_depths × 2 min_samples_leaf` = **6** | **120** | `150` trees (`18,000` trees total) |
| `run_feature_ablation` | `5` subsets × `1` seed (`42`) | **5** | `150` boosting iters |
| **Total** | **48 fits / (task, seed)** | **965 model fits** | — |

### Why It Stalled / Took Too Long
1. **965 full fits (480 on `200,329 × 33` rows):** Half of all fits run on the Full-Stream training set (`200,329` rows). `RF_tuned` (`max_depth=None`, `150` trees) and `HistGB_tuned` (`4` classes × `150` iterations) are the heaviest configurations.
2. **CPU oversubscription (`n_jobs=-1`):** Using `-1` spawns workers on 100% of logical cores on Windows, causing memory/cache contention across `150` trees on `200k` rows and making the PC unresponsive.
3. **Zero progress logging:** `run_stage3b6()` prints nothing until all 965 fits finish, making it impossible to tell if the process is making progress or hung.
4. **Zero checkpointing:** All JSON/CSV outputs are written only at the very end of `run_stage3b6()` (lines 555–570), so `KeyboardInterrupt` loses 100% of completed work.

---

## 2. Assessment of Existing `--fast` Mode & Partial Outputs

- **Existing `--fast` mode:** **Suitable only for a 10-second smoke test, NOT for final results.** It collapses every grid to 1 hardcoded config (`n_estimators=40`, `class_weight=None`, `oversample_cap=20`), which disables validation tuning and breaks `RF_train_oversampled` (`cap=20` < `1,644` `intrusion` rows).
- **Existing partial outputs:** **None can be reused.** Because `run_stage3b6()` writes outputs only after all tasks/models/seeds finish, no partial files were written to `results/stage_3b6_improvements/` before `KeyboardInterrupt`.

---

## 3. Safe Optimization Recommendations (Zero Scientific Compromise)

All 4 MQTT-UAD tasks, all 5 seeds (`42..46`), the full hyperparameter grids, validation-only selection, and training-only oversampling can be **preserved 100% intact** with these engineering improvements:

1. **Atomic Per-`(task_id, model, seed)` Checkpointing + `--resume`:**
   - Save each completed `(task_id, model, seed)` result (including `metrics`, `thresh_info`, and `elapsed_s`) to `results/stage_3b6_improvements/checkpoints/{task_id}__{model}__seed{seed}.json` via write-to-`.tmp` + `os.replace()`.
   - On `--resume`, load existing valid checkpoint files and skip already-completed `(task_id, model, seed)` runs without duplicating or overwriting them.
2. **Safe Bounded CPU Parallelism (`--n-jobs`, default `max(1, (os.cpu_count() or 4) - 2)`):**
   - Leaves 2 logical cores free for Windows OS responsiveness and reduces L3 cache thrashing.
3. **Per-Model / Per-Seed Progress & Elapsed-Time Logging:**
   - Print `[task_id | model | seed X] fit 1/18 ... done in 14.2s (best val macro-F1=0.6842)` with `flush=True` so progress and ETA are always visible.
4. **Load `mqtt_uad_features.csv` Once (Minor I/O Cleanup):**
   - Cache the loaded `SplitDataset` in memory (or load once and derive `binary` / `mqtt_only` views) instead of parsing `mqtt_uad_features.csv` 4 times.
5. **Optional `--tasks` / `--families` CLI Filters:**
   - Allow running one task or one model family at a time (e.g. `--tasks mqtt_uad_full_multiclass`) and combining checkpoints cleanly, so long runs can be completed in manageable chunks.

---

## 4. Exact Proposed Code Changes (Awaiting Approval)

In `ids/train_baselines_improved.py` only (no changes to `datasets/`, `ids/datasets_real.py`, `results/stage_3b_baselines/`, or FSM/simulation/diagnosis modules):
1. Add `_default_n_jobs()` helper (`max(1, (os.cpu_count() or 4) - 2)`), `--n-jobs`, `--resume`, `--tasks`, and `--families` CLI flags.
2. Add `_checkpoint_path(out_dir, task_id, model, seed)` and atomic `_save_json_atomic(path, payload)` helper.
3. Inside `run_stage3b6()`: check for existing checkpoint when `resume=True`; otherwise run `fit_select_and_eval`, print elapsed time + validation macro-F1, and atomically write the checkpoint immediately.
4. Load `mqtt_uad_features.csv` once for `mqtt_only=False` and `mqtt_only=True` and derive binary labels from the multiclass `SplitDataset` without re-reading the CSV.
5. Note: When `--tasks` or `--families` filters are used with `--resume`, rebuild `aggregated_metrics.json`, `model_comparison.csv`, and `STAGE3B6_BASELINE_IMPROVEMENT_REPORT.md` from all available checkpoints in `checkpoints/`.

Awaiting your approval to apply these safe optimizations to `ids/train_baselines_improved.py`.

