# Stage 3B.6 — Pre-Execution Code and Experimental Integrity Audit

> **Audited Files:** `ids/train_baselines_improved.py` (577 lines), `tests/test_stage3b6_improvements.py`, Stage 3A `ids/datasets_real.py`, `datasets/processed/mqtt_uad/mqtt_uad_manifest.json`, and immutable Stage 3B outputs under `results/stage_3b_baselines/`.  
> **Final Decision:** **NOT READY TO RUN** (2 minor reporting/logging gaps in `ids/train_baselines_improved.py` must be fixed before running experiments; zero data-leakage bugs found).

---

## 1. Section-by-Section Audit Scorecard

| Section | Requirement | Status | Summary |
|---|---|---|---|
| **1** | Dataset and split integrity | **PASS** | Uses `load_mqtt_uad()` for all 4 MQTT-UAD tasks; `train`/`val`/`test` remain strictly separated; test data never enters tuning; all 5 seeds (`42..46`) recorded. |
| **2** | Model tuning | **PASS** | `RF_tuned`, `ExtraTrees_tuned`, `HistGB_tuned`, and `RF_train_oversampled` all fit on `X_train` and select `best_cfg` exclusively via validation `macro_f1`; `selected_config` is saved per seed/task. |
| **3** | Training-only oversampling | **PASS** | `cap_oversample_train` (lines 62–76) operates solely on `ds.X_train, ds.y_train`, caps minority classes at `10,000` without downsampling majority classes (`>= target_cap` kept intact), leaves `val`/`test` untouched, and does not use SMOTE. |
| **4** | Binary threshold tuning | **FAIL (Minor)** | Threshold grid `0.10..0.90` (step `0.05`) on `val` positive-class (`attack`, index `1`) `macro_f1` is leakage-free (`lines 79–91, 253–269`), **but** `thresh_info` omits `default_0.5_attack_f1`, `tuned_attack_f1`, `default_0.5_confusion_matrix`, and `tuned_confusion_matrix`. |
| **5** | Feature ablation | **PASS** | All 5 subsets (`all_33_features`, `no_layer_flags`, `no_mitm_nan_cols`, `packet_and_layer_only`, `mqtt_features_only`) use only the 33 manifest-approved features (`lines 282–293`), with zero leakage columns. |
| **6** | Metrics and reporting | **FAIL (Minor)** | `generate_improvement_report` (`lines 378–385`) reads only `model_comparison.csv` instead of `results/stage_3b_baselines/aggregated_metrics.json`, leaving Stage 3B `mitm` F1, `intrusion` F1, and `Balanced Acc` as `—`, and omits a full per-class P/R/F1/support table in the Markdown report. |
| **7** | Scientific limitations | **PASS** | Explicitly discloses within-capture chronological evaluation, no unseen-capture generalization, small `mitm`/`intrusion` test supports (`13` `mitm` in `MQTT-Only`), and no formal fault localization (`lines 424–429`). |

---

## 2. Exact Issues & File/Line References

1. **Missing Confusion Matrices and Attack F1 in Binary Threshold Tuning Output**
   - **Location:** `ids/train_baselines_improved.py`, lines `258–269` (`fit_select_and_eval`) and lines `495–505` (`run_stage3b6`).
   - **Issue:** `m_default` and `m_tuned` both compute `confusion_matrix` and `per_class["attack"]["f1"]`, but `thresh_info` only stores `macro_f1`, `precision`, and `recall`—omitting `default_0.5_attack_f1`, `tuned_attack_f1`, `default_0.5_confusion_matrix`, and `tuned_confusion_matrix` from `threshold_tuning.json`.
2. **Incomplete Stage 3B Baseline Comparison & Missing Per-Class Table in Markdown Report**
   - **Location:** `ids/train_baselines_improved.py`, lines `378–385` (`generate_improvement_report`).
   - **Issue:** Reading only `results/stage_3b_baselines/model_comparison.csv` leaves Stage 3B's `Balanced Acc`, `mitm F1`, and `intrusion F1` as `—` in the Markdown comparison table, even though `results/stage_3b_baselines/aggregated_metrics.json` contains all per-class F1 scores and confusion matrices. In addition, the generated Markdown report lacks a per-class Precision/Recall/F1/Support section for the Stage 3B.6 models (though those metrics are saved in `aggregated_metrics.json`).

---

## 3. Required Corrections Before Execution

1. **In `ids/train_baselines_improved.py` (`fit_select_and_eval` & `run_stage3b6`):**
   - Add `default_0.5_attack_f1`, `tuned_attack_f1`, `default_0.5_confusion_matrix`, and `tuned_confusion_matrix` to `thresh_info`, and aggregate/save the summed confusion matrices for both `0.50` and `tau*` in `threshold_tuning.json`.
2. **In `ids/train_baselines_improved.py` (`generate_improvement_report`):**
   - Load `results/stage_3b_baselines/aggregated_metrics.json` (when present) to populate Stage 3B's `mitm F1`, `intrusion F1`, and `Balanced Acc` (computed from `confusion_matrix_sum`) in the comparison table.
   - Add a **Per-Class Test Metrics** section (Precision, Recall, F1, Support) for all Stage 3B.6 models in `docs/STAGE3B6_BASELINE_IMPROVEMENT_REPORT.md`.

---

## 4. Final Decision

**NOT READY TO RUN** — No data-leakage or split-integrity defects exist, but the two reporting omissions above should be fixed in `ids/train_baselines_improved.py` before executing the full 5-seed experiment. Awaiting your approval to apply these two edits.

