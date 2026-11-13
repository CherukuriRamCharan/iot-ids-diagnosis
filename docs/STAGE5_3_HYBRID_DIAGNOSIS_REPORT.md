# Stage 5.3 Report: Two-Tier Hybrid Deep Learning + Formal TFSM Diagnosis Engine

> **Project:** *Attack Diagnosis and Localization in IoT Intrusion Detection Using Formal Modeling and Deep Learning*  
> **Component:** Stage 5.3 — Hybrid Diagnosis Integration (Statistical BiLSTM Inference + Deterministic TFSM Conformance Adjudication)  
> **Module Created:** `diagnosis_loc/hybrid_diagnosis.py`  
> **Unit Test Suite:** `tests/test_stage5_hybrid.py`  
> **Selected Checkpoint:** `results/stage_5_hybrid/checkpoints/bilstm_multiclass_seed42.pt`

---

## 1. System Architecture

The hybrid diagnostic system couples a **statistical deep learning sequence classifier (Tier 1)** with a **deterministic Timed Finite State Machine conformance engine (Tier 2)**. 

Neither tier overwrites, suppresses, or synthesizes evidence for the other. Each tier operates on its own computational paradigm, and an explicit **Adjudication Arbiter** synthesizes their independent findings into a unified operational alert.

```
                           Observable Event Stream (X)
                                       |
          +----------------------------+----------------------------+
          |                                                         |
          v                                                         v
   [ Tier 1: Statistical DL ]                                [ Tier 2: Formal TFSM ]
   Chronological Sliding Windows (L=16)                     Sensor-Isolated Monitors
   BiLSTM Sequence Classifier                                Timed State Machine Checker
          |                                                         |
          v                                                         v
   Empirical Pattern Evidence:                               Formal Mathematical Evidence:
   - Predicted Attack Category                               - Formal Conformance: PASS / FAIL
   - Raw Softmax Confidence Score                            - Physical Deviation Type (e.g. TIMING_VIOLATION)
   - Multi-Step Temporal Context                             - Suspected Transition (e.g. t21) & State (s9)
                                                             - Candidate Mutant Hypotheses (e.g. DFA_t21)
          |                                                         |
          +----------------------------+----------------------------+
                                       |
                                       v
                    [ Hybrid Adjudication Arbiter ]
                    - Cross-examines statistical & formal evidence
                    - Reconciles consensus, violations, & anomalies
                    - Separates empirical heuristics from math proofs
                                       |
                                       v
                        Unified Hybrid Diagnosis Report
```

---

## 2. Production Model Checkpoint Selection

In strict compliance with Requirement 2, **model selection was performed exclusively using validation split performance (sessions 14–18)** without inspecting test split outcomes:

1. **Validation Macro-F1 Comparison:**
   - **`BiLSTMClassifier` (Validation Macro-F1):** **$0.9604 \pm 0.0152$**
   - **`LSTMClassifier` (Validation Macro-F1):** $0.9412 \pm 0.0210$
2. **Contextual Encoding Superiority:**
   - Because 16-packet sequence windows interleave events from four concurrent sensors, the bidirectional recurrent pass in BiLSTM captures both preceding and succeeding temporal context (e.g., normal traffic resuming after a delayed packet burst), producing consistently higher validation Macro-F1.
3. **Selected Production Checkpoint:**
   - Architecture: **BiLSTM** (Seed 42)
   - Configuration: $14$ input features, $64$ hidden units, $2$ recurrent layers, dropout $0.20$, $7$ output classes.
   - Validation Macro-F1: **$0.9782$**
   - Paths: `results/stage_5_hybrid/checkpoints/bilstm_multiclass_seed42.pt` and `.json`.

---

## 3. Live Inference Workflow & Anti-Leakage Safeguards

1. **Event Ingestion & Sanitization:**
   - Events are accepted as dictionaries or `UnifiedEvent` instances.
   - Quarantined ground-truth fields (`is_attack`, `attack_type`, `attack_target`, `fault_location`, `original_payload`, `original_timestamp`) and simulator watermarks (`tampered_by`, `is_event_flood`, `has_tamper_flag`) are strictly stripped before processing.
   - Events are sorted chronologically by `timestamp` and `sequence_number`.
2. **Feature Extraction ($F=14$):**
   - Extracts strictly the 14 approved observable features: `inter_arrival`, `payload_size`, `qos`, `seq_delta`, 4 sensor topic indicators, 4 event type indicators, `numeric_reading` (with $-1.0$ sentinel), and `has_alert_flag`.
3. **Standardization:**
   - Transforms the feature matrix using `MaskedStandardScaler` parameters (`mean` and `scale`) saved from the training split.
   - Binary indicator columns are preserved untouched ($0.0$ or $1.0$).
4. **Sliding Window Construction:**
   - If total events $N < 16$, the engine reports `INSUFFICIENT_SEQUENCE` while allowing formal FSM evaluation to proceed unimpeded.
   - If $N \ge 16$, constructs $N - 16 + 1$ overlapping windows of shape $(16, 14)$.
5. **Inference & Scoring:**
   - Evaluates batches through `BiLSTMClassifier`, computing softmax scores.
   - The primary predicted class and maximum softmax value are reported as the **model confidence score** (explicitly documented as an empirical heuristic, not a calibrated Bayesian posterior).

---

## 4. Formal TFSM Diagnosis Integration

Tier 2 executes `UnifiedDiagnosisEngine` directly on the event stream without modification:
- Maintains independent `SensorMonitor` state machines for `motion`, `temperature`, `ultrasonic`, and `rfid`.
- Checks sensor-isolated elapsed time against keepalive thresholds (`TIMEOUT_THRESHOLDS`).
- Evaluates transition input/output conformance against nominal TFSM definitions (`_on_input`).
- Checks duplicate sequence numbers (`REPLAY_DUPLICATE`) vs. sequence regressions (`SEQUENCE_REGRESSION`).
- Inspects physical domain limits (e.g., temperature in $[-20^\circ\text{C}, 70^\circ\text{C}]$).
- **Non-Interference Guarantee:** DL predictions never synthesize observed outputs, never alter state transitions, and never fabricate candidate mutants.

---

## 5. Four Explicit Adjudication Outcomes

The Adjudication Arbiter cross-examines Tier 1 and Tier 2 outputs to assign one of four primary outcomes (plus two operational boundary states):

| Adjudication Outcome | Formal FSM Status | DL Prediction | Interpretation & Operational Action |
|---|:---:|:---:|---|
| **`CONSENSUS`** | **Anomaly Detected** | **Attack Predicted** | **High-Confidence Validated Alert.** Both statistical temporal patterns and mathematical state/timing invariants detect an attack. Tier 1 provides attack family classification; Tier 2 provides exact transition and candidate mutant proof. |
| **`FORMAL_VIOLATION_DL_NORMAL`** | **Anomaly Detected** | **Normal** | **Formal Violation Override.** The formal monitor proves a hard protocol, sequence, or timing breach that the statistical model missed. The formal violation is strictly preserved, flagging a DL false negative. |
| **`STATISTICAL_ANOMALY`** | **Nominal (PASS)** | **Attack Predicted** | **Statistical Warning for Inspection.** The neural network flags distributional or temporal anomalies, but state transitions remain valid under nominal TFSM rules. Flagged for operator review; **no formal proof of attack is claimed**. |
| **`NOMINAL`** | **Nominal (PASS)** | **Normal** | **Verified Benign Operation.** Both tiers independently verify normal physical sensor operation. |
| *`INSUFFICIENT_SEQUENCE`* | Monitored | N/A ($N < 16$) | Fewer than 16 packets available for sequence windowing. Formal FSM runs independently; formal anomalies are **never suppressed**. |
| *`MODEL_UNAVAILABLE`* | Monitored | N/A (Missing) | DL model checkpoint missing or failed to load. Formal FSM runs independently; formal anomalies are **never suppressed**. |

---

## 6. Example Diagnostic Reports

### Example 1: DoS Flood Attack (`CONSENSUS`)
```json
{
  "hybrid_outcome": "CONSENSUS",
  "outcome_explanation": "CONSENSUS: Both diagnostic tiers concur. Formal TFSM detected an anomaly (DoS, deviation=BURST_FLOOD) and deep learning sequence model classified an attack as 'DoS' with model confidence score 0.9982.",
  "agreement": true,
  "total_events": 45,
  "dl_attack_detected": true,
  "dl_predicted_category": "DoS",
  "dl_confidence_score": 0.9982,
  "dl_window_count": 30,
  "formal_anomaly_detected": true,
  "formal_conformance": false,
  "formal_primary_attack": "DoS",
  "affected_sensors": ["motion"],
  "suspected_sensor": "motion",
  "deviation_type": "BURST_FLOOD",
  "candidate_fault_mutants": ["dos_queue_exhaustion"],
  "formal_evidence": [
    "High-frequency burst rate on motion: dt=0.0050s",
    "Output mismatch on transition t8 from state s5: expected o4, observed o6"
  ]
}
```

### Example 2: Formal Conformance Violation with DL False Negative (`FORMAL_VIOLATION_DL_NORMAL`)
```json
{
  "hybrid_outcome": "FORMAL_VIOLATION_DL_NORMAL",
  "outcome_explanation": "FORMAL VIOLATION (DL DISAGREEMENT): Formal TFSM detected a hard protocol, timing, or sequence deviation (OUTPUT_MISMATCH, primary=DataFalsification) on sensor 'temperature', but deep learning sequence model predicted Normal with confidence 0.9214. Formal mathematical violation proof is strictly preserved.",
  "agreement": false,
  "total_events": 25,
  "dl_attack_detected": false,
  "dl_predicted_category": "Normal",
  "dl_confidence_score": 0.9214,
  "formal_anomaly_detected": true,
  "formal_conformance": false,
  "formal_primary_attack": "DataFalsification",
  "affected_sensors": ["temperature"],
  "suspected_sensor": "temperature",
  "suspected_transition": "t21",
  "suspected_state": "s9",
  "expected_output": "o11",
  "observed_output": "o6",
  "deviation_type": "OUTPUT_MISMATCH",
  "candidate_fault_mutants": ["DFA_t21", "dfa_sensor_falsification"],
  "formal_evidence": [
    "Output mismatch on transition t21 from state s9: expected o11, observed o6"
  ]
}
```

### Example 3: Statistical Drift Without State Machine Violation (`STATISTICAL_ANOMALY`)
```json
{
  "hybrid_outcome": "STATISTICAL_ANOMALY",
  "outcome_explanation": "STATISTICAL ANOMALY: Deep learning sequence model detected temporal patterns resembling 'DoS' with model confidence score 0.8845, but all observable events nominally conform to formal TFSM transitions and timing bounds. Flagged for operator inspection; no formal proof of an attack.",
  "agreement": false,
  "total_events": 30,
  "dl_attack_detected": true,
  "dl_predicted_category": "DoS",
  "dl_confidence_score": 0.8845,
  "formal_anomaly_detected": false,
  "formal_conformance": true,
  "formal_primary_attack": "Normal",
  "affected_sensors": [],
  "candidate_fault_mutants": [],
  "formal_evidence": []
}
```

---

## 7. Unit Test Suite Verification (`tests/test_stage5_hybrid.py`)

A dedicated suite of 14 unit tests was created to verify all boundary conditions, feature ordering, and adjudication branches without retraining models:

| # | Test Method Name | Verification Scope | Status |
|:---:|---|---|:---:|
| 1 | `test_1_successful_model_loading` | Model weights, metadata, and scaler parameters load from checkpoint. | **PASS** |
| 2 | `test_2_model_input_feature_ordering` | Verifies feature list matches `SELECTED_OBSERVABLE_FEATURES` exactly ($F=14$). | **PASS** |
| 3 | `test_3_correct_16_event_window_construction` | Verifies $20$ events construct exactly $5$ windows ($20 - 16 + 1$). | **PASS** |
| 4 | `test_4_session_and_sensor_isolation` | All 4 sensors maintain isolated state and timing monitors. | **PASS** |
| 5 | `test_5_consensus_outcome` | Both tiers flag attack $\to$ `CONSENSUS`, `agreement=True`. | **PASS** |
| 6 | `test_6_formal_violation_with_dl_predicting_normal` | Formal flags violation, DL predicts Normal $\to$ `FORMAL_VIOLATION_DL_NORMAL`. | **PASS** |
| 7 | `test_7_statistical_anomaly_with_formal_conformance_passing` | Formal passes, DL predicts attack $\to$ `STATISTICAL_ANOMALY`, no false proof. | **PASS** |
| 8 | `test_8_nominal_outcome` | Both tiers verify normal traffic $\to$ `NOMINAL`, `agreement=True`. | **PASS** |
| 9 | `test_9_insufficient_sequence_handling` | $10$ events ($<16$) $\to$ `INSUFFICIENT_SEQUENCE`, formal runs independently. | **PASS** |
| 10 | `test_10_model_unavailable_handling` | Missing checkpoint $\to$ `MODEL_UNAVAILABLE`, formal anomaly **not suppressed**. | **PASS** |
| 11 | `test_11_formal_evidence_preservation` | Candidate mutants, deviation types, and transitions preserved verbatim. | **PASS** |
| 12 | `test_12_no_ground_truth_metadata_leakage` | Injected metadata and labels are stripped before feature extraction. | **PASS** |
| 13 | `test_13_no_fabricated_observed_fsm_outputs` | DL predictions never alter or synthesize formal FSM outputs. | **PASS** |
| 14 | `test_14_deterministic_inference` | Repeated diagnosis calls produce bitwise identical hybrid reports. | **PASS** |

---

## 8. Critical Boundaries & Limitations

1. **Software Simulation Boundary:**
   - Sensor events are generated via discrete-event software simulation with controlled timing jitter.
   - Real-world wireless sensor deployments exhibit physical RF fading, WiFi CSMA/CA backoff collisions, and operating system scheduling jitter.
   - The hybrid architecture explicitly mitigates this: even if distributional shift degrades DL statistical pattern recognition, the formal TFSM operates on invariant transition graphs and timeout boundaries.
2. **Confidence Scores are Uncalibrated Heuristics:**
   - Raw softmax outputs from deep recurrent networks are empirical confidence metrics, not Bayesian probabilities.
3. **Absence of Malicious Intent Proof:**
   - The system detects **observable protocol, timing, and payload deviations**. It cannot infer adversary intent; a benign network interruption or sensor hardware failure that violates keepalive thresholds is formally reported as a `TIMING_VIOLATION` (matching mutants `BD_s8`, `BD_s9`).

---

## 9. Verification Commands (Windows CMD)

To verify the Stage 5.3 hybrid engine and the entire cumulative test inventory in Command Prompt:

```cmd
cd /d C:\Users\cheru\Downloads\iot-ids-diagnosis\iot-ids-diagnosis

:: 1. Run Stage 5.3 hybrid diagnosis unit tests (14 tests)
python -m unittest tests.test_stage5_hybrid -v

:: 2. Run cumulative repository test suite (125 tests across 13 test files)
python -m unittest discover -s tests -v

:: 3. Confirm formal TFSM mutation score remains 100% (59/59 mutants killed)
python -m models_fsm.verify
```

#### Cumulative Test Inventory (125 Tests across 13 Files)
- `tests.test_fsm_sensors`: 21 tests
- `tests.test_unified_simulator`: 8 tests
- `tests.test_attack_injector`: 8 tests
- `tests.test_phase3_data`: 10 tests
- `tests.test_stage3_dataset`: 9 tests
- `tests.test_stage3b_baselines`: 3 tests
- `tests.test_stage3b6_improvements`: 3 tests
- `tests.test_stage3b6_runtime`: 5 tests
- `tests.test_stage3c_sequences`: 8 tests
- `tests.test_unified_diagnosis`: 16 tests
- `tests.test_stage5_dataset`: 12 tests
- `tests.test_stage5_training`: 8 tests
- `tests.test_stage5_hybrid`: 14 tests
- **Total: 125 / 125 tests passing (100% pass rate)**.

