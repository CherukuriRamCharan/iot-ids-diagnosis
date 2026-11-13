# Phase 3 Implementation Plan: Model Development, Attack Diagnosis, and Formal Localization

> **Project:** Attack Diagnosis and Localization in IoT Intrusion Detection Using Formal Modeling and Deep Learning  
> **Prerequisites:** Phase 2 (`OVERALL PHASE 2 VALIDATION PASSED: True`) and Phase 2.5 (`READY FOR PHASE 3 (WITH CONDITIONS)`)  
> **Status:** Planning only — no models trained and no existing code or datasets modified.

---

## 1. Architectural Separation: Classification vs. Diagnosis vs. Formal Localization

To maintain strict scientific integrity (per `docs/PHASE2_INDEPENDENT_REVIEW.md`), Phase 3 separates three distinct capabilities:

1. **Attack Detection & Classification (ML / DL Models on `MQTT-UAD` & `MQTTset`)**:
   - *Binary Detection*: `normal` vs. `attack`.
   - *Multiclass Classification*: 4-class `type` (`normal`, `DoS`, `mitm`, `intrusion`) on **MQTT-UAD** (primary real-packet dataset), and 6-class `Cat` (`Normal`, `Flood`, `MalariaDoS`, `Bruteforce`, `Malformed`, `Slowite`) on **MQTTset** (separate secondary flow-level benchmark).
2. **Attack Diagnosis (Evidence-Backed Alerting Module)**:
   - Explains *why* a prediction was made by combining: (a) predicted class + calibrated confidence, (b) top contributing behavioral features (`frame.time_delta`, `has_ip`, `mqtt.msgtype`, `mqtt.len`, `mqtt.qos`, `mqtt.dupflag`), (c) post-hoc metadata attribution joined via `(source_file, source_row_idx)` from `mqtt_uad_metadata.csv` (`ip.src`, `ip.dst`, `eth.src`, `eth.dst`, `mqtt.topic`, `mqtt.clientid`, `mqtt.msg` where populated), and (d) explicit evidence-limitation flags (e.g., Layer-2 ARP `mitm` with missing IP/MQTT payload).
   - *Note:* `*_metadata.csv` columns are **never** fed into model training/inference `X`.
3. **Formal Fault Localization (`models_fsm/`, `simulation/`, `diagnosis_loc/`)**:
   - **Public datasets (`MQTT-UAD`, `MQTTset`) do NOT contain formal TFSM state/transition fault-localization ground truth.**
   - Formal fault localization remains anchored in the project's reconstructed TFSM + mutation-testing pipeline (`models_fsm/motion_sensor.py`, `simulation/simulator.py`, `ids/train.py`, `diagnosis_loc/report.py`), which evaluates `TwoHeadLSTM` across independent simulated network-noise sessions (`train 0-6`, `val 7`, `test 8-9`) and flags diagnosis/localization consistency.
   - Phase 3 will add a **protocol-to-TFSM diagnostic bridge** that maps observed MQTT-UAD behavioral signatures (e.g., anomalous `CONNECT`/`DISCONNECT`/`PUBLISH` sequences or timing violations) to candidate TFSM mutant families without fabricating ground-truth localization labels on public captures.

---

## 2. Model Development Plan & Mandatory Phase 2.5 Guards

### 2.1 Models to Implement
All models will be trained and evaluated **separately** on **MQTT-UAD** (primary) and **MQTTset** (secondary benchmark):

| Family | Model | Justification | Sentinel (`-1.0`) & Scaling Rule |
|---|---|---|---|
| **Traditional ML (Primary)** | `RandomForestClassifier` (`scikit-learn`) | Strong non-linear tabular baseline; natively handles `-1.0` sentinels and `has_mqtt*` flags; interpretable feature importances. | Raw `33` (MQTT-UAD) / `62` (MQTTset) features; `class_weight="balanced"`. |
| **Traditional ML (Additional)** | `DecisionTreeClassifier` (`scikit-learn`) | Interpretable rule-path baseline for attack diagnosis transparency; already supported by `requirements.txt` (`scikit-learn`) with zero new dependencies. | Same as Random Forest; `max_depth` tuned on `val` macro-F1. |
| **Deep Learning (Sequence)** | **LSTM** (`torch.nn.LSTM`) | Captures temporal packet bursts (`DoS`) and multi-packet protocol sequences (`intrusion`) on MQTT-UAD; matches the project's existing `ids/` PyTorch stack. | **Train-only z-score normalization** on non-sentinel (`!= -1.0`) values; set sentinel entries to `0.0` post-scaling; retain `has_mqtt*` binary indicators. |
| **Deep Learning (Attention)** | **Transformer Encoder** (`torch.nn.TransformerEncoder`) | Captures long-range inter-packet dependencies and cross-feature interactions (tabular/windowed attention) for direct comparison against LSTM. | Same train-only masked normalization as LSTM. |

### 2.2 Mandatory Leakage & Integrity Guards (From Phase 2.5)
1. **Strict Within-Split Sequence Windowing (MQTT-UAD)**:
   - Because `train`, `val`, and `test` meet at zero-gap chronological boundaries, sliding windows of length `W` (e.g., `W = 16` or `32` packets) **must** be constructed **independently inside each `(source_file, split)` partition**—never across split boundaries.
   - Window label: assign the label of the final packet (or majority/any-attack rule, documented consistently) and verify zero row-index overlap across splits.
2. **MQTT-Only Subset Ablation (MQTT-UAD)**:
   - Evaluate models on both the **full packet stream** (`286,186` rows) and the **MQTT-only subset** (`has_mqtt == 1`) to verify models are learning MQTT attack behavior rather than just `"MQTT vs. non-MQTT background traffic"`.
3. **Imbalance Handling**:
   - Use inverse-frequency / balanced class weights in loss functions (`CrossEntropyLoss(weight=...)` for PyTorch, `class_weight="balanced"` for scikit-learn) to prevent ignoring `intrusion` (0.66%), `mitm` (1.35%), and `Slowite` (0.69%).
4. **No Metadata Leakage**:
   - Assert at data-load time that `source_file`, `source_row_idx`, `split`, `type`/`label`/`Cat`, and all `*_metadata.csv` columns are excluded from `X`.

---

## 3. Evaluation Protocol

Every model on both datasets will be evaluated over **5 fixed random seeds** (matching `ids/train.py` style, e.g., `seeds = [42, 43, 44, 45, 46]`, reporting `mean ± std`) and logged to `results/`:

- **Metrics**: Accuracy, **Macro-Precision**, **Macro-Recall**, **Macro-F1** (primary selection metric), Weighted-F1, **Per-Class Precision/Recall/F1**, **PR-AUC** (for binary & one-vs-rest rare classes), and **Normalized Confusion Matrices** (PNG + JSON).
- **Efficiency**: Wall-clock training time (seconds) and inference latency (ms per 1,000 samples/windows).
- **Explicit Evaluation Caveats**:
  - **MQTT-UAD** results are reported strictly as **within-capture chronological holdout** performance (not unseen-network generalization).
  - **MQTTset** results are reported strictly as a **secondary `Flow_ID`-grouped deduplicated benchmark** (noting the `Normal` deduplication prior shift from `81.45%` to `17.81%` and single-capture-per-`Cat` artifact).
  - **TFSM Simulation** (`ids/train.py`) continues to be reported separately for **unseen-session diagnosis and formal fault localization** (including diagnosis/localization consistency).

---

## 4. Staged Implementation Roadmap (Stages 3A–3F)

| Stage | Scope | Files to Create / Modify (New Modules Only) | Expected Outputs & Tests | Completion Criteria |
|---|---|---|---|---|
| **3A** | **Training Data & Feature Pipeline Verification** | `ids/datasets_real.py` (dataset loader, train-only masked scaler, within-split window builder), `tests/test_phase3_data.py` | Unit tests verifying `0` cross-split windows, `0` metadata columns in `X`, train-only scaler stats, and `has_mqtt == 1` filter. | All unit tests pass; `X` shapes & class weights verified on both datasets. |
| **3B** | **Traditional ML Baselines** | `ids/train_baselines.py` (`RandomForestClassifier`, `DecisionTreeClassifier`, binary + multiclass, MQTT-UAD + MQTTset) | `results/baselines/` (metrics JSON, confusion matrices, feature importances, timing logs). | 5-seed runs complete on MQTT-UAD (full + MQTT-only) and MQTTset. |
| **3C** | **Deep Learning: LSTM** | `ids/models_dl.py` (`PacketLSTM`), `ids/train_dl.py` | `results/lstm/` (training curves, 5-seed test metrics, confusion matrices, latency). | LSTM trained & evaluated cleanly on within-split windows (MQTT-UAD) and flows (MQTTset). |
| **3D** | **Deep Learning: Transformer** | `ids/models_dl.py` (`PacketTransformer`), `ids/train_dl.py` | `results/transformer/` (5-seed test metrics, confusion matrices, latency). | Transformer trained & evaluated under identical splits/metrics as LSTM. |
| **3E** | **Attack Diagnosis & TFSM Integration** | `diagnosis_loc/real_diagnoser.py` (joins predictions + `*_metadata.csv` + confidence/limitation flags + TFSM anomaly mapping), `tests/test_diagnoser.py` | Sample structured diagnostic alert reports in `results/diagnosis_alerts_mqtt_uad.md`. | Alerts include class, confidence, packet evidence, metadata attribution, and explicit limitation tags. |
| **3F** | **Evaluation & Comparison** | `ids/compare_models.py`, `docs/PHASE3_RESULTS_REPORT.md` | Unified comparison tables (RF vs. DT vs. LSTM vs. Transformer) kept strictly separate for MQTT-UAD, MQTTset, and TFSM Simulation. | All metrics, confusion matrices, ablations, and limitations documented without overclaiming. |

---

## 5. Exact First Implementation Task (Awaiting Approval)

**First Task — Stage 3A (`ids/datasets_real.py` + `tests/test_phase3_data.py`):**  
Implement the leakage-safe data loader for `datasets/processed/mqtt_uad/` and `datasets/processed/mqttset/`, including:
1. Assertion-checked feature/metadata separation.
2. Train-only masked normalization for `-1.0` sentinels (for LSTM/Transformer).
3. Strictly within-split sliding-window sequence construction for MQTT-UAD (`train`, `val`, `test` windowed independently).
4. Automated verification tests (`tests/test_phase3_data.py`).

**Please confirm if you approve starting Stage 3A.**

