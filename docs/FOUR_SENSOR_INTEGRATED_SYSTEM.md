# Comprehensive System Architecture & Specification: Four-Sensor IoT Intrusion Detection, Formal Diagnosis & Fault Localization

> **Project:** *Attack Diagnosis and Localization in IoT Intrusion Detection Using Formal Modeling and Deep Learning*  
> **Repository:** `iot-ids-diagnosis`  
> **System Baseline:** Four Software-Simulated Sensors (Motion, Temperature, Ultrasonic, RFID)  
> **Verification Status:** Stages 1–6 Complete; 59/59 Mutants Killed; 144/144 Unit Tests Passing

---

## 1. Executive Summary & Architectural Overview

This system provides a mathematically grounded, two-tier intrusion detection and fault localization framework for multi-sensor IoT networks. The architecture pairs **statistical deep sequence modeling (Tier 1: BiLSTM)** with **deterministic Timed Finite State Machine conformance verification (Tier 2: TFSM)**:

1. **Statistical Tier (Tier 1):** Captures high-dimensional, cross-packet temporal dynamics across sliding 16-packet observation windows to detect distributed and multi-step attack patterns.
2. **Formal Tier (Tier 2):** Enforces strict sensor-isolated timing bounds, protocol transition validities, sequence monotonicity, and physical plausibility invariants to mathematically verify conformance and pinpoint root-cause mutant faults.
3. **Hybrid Adjudication Arbiter:** Reconciles the outputs of both tiers into an explicit, transparent operational verdict (`CONSENSUS`, `FORMAL_VIOLATION_DL_NORMAL`, `STATISTICAL_ANOMALY`, `NOMINAL`) while strictly preserving independent evidence fields without cross-contamination.

```
                              Observable MQTT Packet Stream (X)
                                              |
               +------------------------------+------------------------------+
               |                                                             |
               v                                                             v
    [ Tier 1: Statistical DL ]                                    [ Tier 2: Formal TFSM ]
    16-Packet Sliding Windows                                     Sensor-Isolated Monitors
    BiLSTM Sequence Classifier                                    Timed State Machine Checker
               |                                                             |
               v                                                             v
    Empirical Pattern Evidence:                                   Formal Mathematical Evidence:
    - Predicted Attack Category                                   - Formal Conformance: PASS / FAIL
    - Softmax Confidence Score                                    - Physical Deviation Type
    - Sliding Window Predictions                                  - Suspected State & Transition
                                                                  - Candidate Fault Mutants
               |                                                             |
               +------------------------------+------------------------------+
                                              |
                                              v
                              [ Hybrid Adjudication Arbiter ]
                              - Cross-examines empirical & formal proofs
                              - Reconciles consensus, violations, & anomalies
                              - Preserves strict evidence separation
                                              |
                                              v
                                Unified Hybrid Diagnosis Report
```

> [!IMPORTANT]
> **Simulation Scope Boundary:** All four IoT sensors (`motion`, `temperature`, `ultrasonic`, `rfid`) are deterministic, software-simulated components executed in Python. No claims of physical ESP32 microcontroller or hardware bench deployment are made.

---

## 2. Formal Timed Finite State Machine (TFSM) Models

Each sensor is formally modeled as a Timed Mealy Machine with finite states $S$, inputs $I$, outputs $O$, transitions $T \subseteq S \times I \times \Phi \times O \times 2^C \times S$, and clock constraints $\Phi$:

### 2.1 Sensor TFSM Specifications

| Sensor | States ($S$) | Primary Input Symbols ($I$) | Primary Output Symbols ($O$) | Key Timing Invariants ($\Phi$) | Attack Mutants Killed | Source Module |
| :--- | :---: | :--- | :--- | :--- | :---: | :--- |
| **Motion** | 6 ($s_1..s_6$) | $i_1$ (init), $i_2$ (poll), $i_3$ (motion_detected), $i_4$ (tamper) | $o_1$ (ack), $o_2$ (idle), $o_3$ (alert), $o_4$ (tamper_alert), $o_6$ (err) | Hold time $\ge 1.0\text{s}$, Heartbeat $\le 5.0\text{s}$ | **14 / 14** | [`models_fsm/motion_sensor.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/models_fsm/motion_sensor.py) |
| **Temperature** | 7 ($s_1..s_7$) | $i_7$ (init), $i_8$ (read), $i_9$ (high_temp), $i_{10}$ (sensor_fault) | $o_7$ (ack), $o_8$ (data), $o_9$ (overheat_alert), $o_{10}$ (fault_flag), $o_6$ (err) | Sampling interval $\Delta t \in [0.8\text{s}, 2.5\text{s}]$, Range: $[-20^\circ\text{C}, 85^\circ\text{C}]$ | **15 / 15** | [`models_fsm/temperature_sensor.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/models_fsm/temperature_sensor.py) |
| **Ultrasonic** | 9 ($s_1..s_9$) | $i_{11}$ (init), $i_{12}$ (trigger_pulse), $i_{13}$ (echo_recv), $i_{14}$ (timeout) | $o_{11}$ (ack), $o_{12}$ (pulse_sent), $o_{13}$ (dist_val), $o_{14}$ (out_of_range), $o_6$ (err) | Pulse-echo $\Delta t \in [0.001\text{s}, 0.038\text{s}]$, Range: $[2\text{cm}, 400\text{cm}]$ | **15 / 15** | [`models_fsm/ultrasonic_sensor.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/models_fsm/ultrasonic_sensor.py) |
| **RFID** | 7 ($s_1..s_7$) | $i_{21}$ (init), $i_{22}$ (card_present), $i_{23}$ (auth_req), $i_{24}$ (read_uid), $i_{25}$ (card_remove) | $o_{21}$ (ack), $o_{22}$ (field_active), $o_{23}$ (auth_ok), $o_{24}$ (uid_data), $o_{25}$ (auth_fail), $o_6$ (err) | Auth challenge timeout $\le 0.5\text{s}$, Anti-collision $\le 0.2\text{s}$ | **15 / 15** | [`models_fsm/rfid_sensor.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/models_fsm/rfid_sensor.py) |
| **Total** | **29** | — | — | — | **59 / 59 (100%)** | [`models_fsm/verify.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/models_fsm/verify.py) |

### 2.2 Formal Mutation Analysis
The formal verification suite synthesizes mutant test cases where transitions, timing bounds, or output functions are systematically perturbed (e.g., $DFA_{t21}$ changes output on transition $t_{21}$, $BD_{s8}$ breaks boundary timing). Every mutant is validated against the nominal specification:
$$\forall m \in \text{MUTANTS}, \quad \text{signature}(\mathcal{M}_m(T_m)) \ne \text{signature}(\mathcal{M}_{\text{nominal}}(T_m))$$
All 59 synthetic mutants are killed by their designated test sequences.

---

## 3. Unified Simulator & Attack Injector Dynamics

The discrete-event simulation engine ([`simulation/unified_simulator.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/simulation/unified_simulator.py)) manages a single virtual timeline advancing in discrete ticks ($\Delta \tau = 0.01\text{s}$). Four independent sensor runtimes run concurrently and publish structured MQTT event dictionaries.

### 3.1 Simulated Attack Taxonomy

An isolated attack injector ([`simulation/attack_injector.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/simulation/attack_injector.py)) interposes attacks at the communication layer without corrupting ground-truth sensor state:

1. **Denial of Service (DoS):** Floods the broker with rapid bursts ($\Delta t \le 0.005\text{s}$) or queue-exhaustion packets targeting a specific sensor.
2. **Replay:** Captures legitimate historical packets and injects them after a transmission delay, producing duplicated sequence numbers and stale timestamps.
3. **Man-in-the-Middle (MITM):** Intercepts in-transit packets and tampers with payload values, causing discrepancies between observed outputs and physical state.
4. **Data Falsification:** Injects out-of-bounds physical sensor readings (e.g., ultrasonic distance $< 2\text{cm}$ or $> 400\text{cm}$, temperature $> 85^\circ\text{C}$).
5. **Delay Attack:** Holds legitimate packets in a simulated buffer and releases them after an excessive timeout ($\Delta t > t_{\text{timeout}}$) without duplicating sequence numbers.
6. **Identity Spoofing:** Injects unauthorized RFID UIDs or spoofed MAC/sensor IDs while attempting credential-gated operations.

---

## 4. Dataset Pipeline & Feature Provenance Audit

The multi-sensor dataset ([`datasets/generated/four_sensors/four_sensors_dataset.csv`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/datasets/generated/four_sensors/four_sensors_dataset.csv)) contains **3,090 simulated events** across **24 independent sessions**.

### 4.1 Strict Session-Level Partitioning
To guarantee zero cross-session data leakage and avoid temporal correlation artifacts:
- **Training Set (Sessions 1–13):** 1,672 events (54.1%)
- **Validation Set (Sessions 14–18):** 644 events (20.8%)
- **Test Set (Sessions 19–24):** 774 events (25.1%)

### 4.2 Feature Provenance Audit & Zero-Leakage Guarantee
Only **14 observable features** available from raw MQTT packets on the network wire are admitted into the input tensor $X$:

```python
SELECTED_OBSERVABLE_FEATURES = [
    "inter_arrival",       # Time elapsed since immediately preceding packet
    "payload_size",       # Size of MQTT payload in bytes
    "qos",                # MQTT Quality of Service flag (0, 1, 2)
    "seq_delta",          # Monotonic sequence delta (seq_t - seq_{t-1})
    "is_motion",          # One-hot sensor identity: Motion
    "is_temperature",     # One-hot sensor identity: Temperature
    "is_ultrasonic",      # One-hot sensor identity: Ultrasonic
    "is_rfid",            # One-hot sensor identity: RFID
    "is_publish",         # One-hot MQTT packet type: PUBLISH
    "is_alert",           # One-hot MQTT packet type: ALERT
    "is_heartbeat",       # One-hot MQTT packet type: HEARTBEAT
    "is_connect",         # One-hot MQTT packet type: CONNECT
    "numeric_reading",    # Parsed physical sensor reading (or 0.0)
    "has_alert_flag",     # Wire-level alert bit in JSON payload
]
```

> [!CAUTION]
> **Quarantined & Excluded Metadata (Zero Leakage):**
> 1. `tampered_by` & `has_tamper_flag`: Simulator injection watermarks. **EXCLUDED.**
> 2. `event_type == "FLOOD"` & `is_event_flood`: Simulator shortcut flag. **EXCLUDED.**
> 3. `observed_state` & `output_symbol`: Internal TFSM state variables. **EXCLUDED.**
> 4. `is_attack`, `attack_type`, `attack_target`, `fault_location`: Ground-truth labels. **EXCLUDED.**

---

## 5. Sequence Deep Learning Architectures & Production Selection

### 5.1 Model Specifications
- **Window Size ($L$):** 16 consecutive events (stride $S = 1$, window strictly intra-session).
- **Labeling Policy:** Multi-class label assigned strictly to the **last event** ($t_{16}$) of each sliding window.
- **Normalization:** `MaskedStandardScaler` fitted **strictly on training sessions 1–13**; continuous features normalized via $z = (x - \mu)/\sigma$, binary one-hot indicators preserved as $\{0.0, 1.0\}$.
- **Loss Function:** Cross-Entropy with inverse-frequency class weighting:
  $$w_c = \frac{N}{K \cdot N_c}$$

### 5.2 Multi-Seed Benchmark Evaluation (Seeds 42–46)

| Architecture | Parameters | Val Macro-F1 (Mean $\pm$ Std) | Test Macro-F1 (Mean $\pm$ Std) | Training Epochs | Early Stop Metric |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **BiLSTMClassifier** | **94,407** | **$0.9604 \pm 0.0152$** | **$0.9488 \pm 0.0189$** | $\le 50$ (Patience: 5) | Val Macro-F1 |
| **LSTMClassifier** | 50,055 | $0.9412 \pm 0.0210$ | $0.9325 \pm 0.0245$ | $\le 50$ (Patience: 5) | Val Macro-F1 |

### 5.3 Production Checkpoint Selection Rationale
In accordance with strict ML evaluation standards, the production model was selected **purely on validation split Macro-F1** (sessions 14–18) without peeking at the test set:
- **Selected Checkpoint:** `BiLSTMClassifier` (Seed 42).
- **Validation Macro-F1:** **$0.9782$**.
- **Rationale:** The bidirectional recurrent layers process preceding and succeeding contextual transitions across interleaved multi-sensor events, providing superior resilience against transient timing jitters.

---

## 6. Formal FSM-Based Diagnosis & Fault Localization Engine

Implemented in [`diagnosis_loc/unified_diagnosis.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/diagnosis_loc/unified_diagnosis.py), the formal engine tracks runtime event streams against the nominal TFSM specification.

### 6.1 Key Diagnostic Invariants
1. **Sensor-Isolated Timing Baselines:** Timing deltas are calculated strictly per-sensor ($\Delta t = t_{\text{curr}} - t_{\text{last}}[sid]$) to prevent false burst anomalies caused by normal multi-sensor interleaving.
2. **Disentangled Sequence Regressions:** Differentiates duplicate sequence arrivals (`REPLAY_DUPLICATE`) from packet reordering caused by buffering delays (`SEQUENCE_REGRESSION`).
3. **Guarded Fault Mutant Attribution:** DFA transition mutants are attributed only to fresh, non-replayed sequences. When a broker emits rejection symbol $o_6$ on a replayed packet, the engine recognizes broker rejection rather than sensor hardware failure.
4. **Physical Bounds Validation:** Enforces sensor-specific ranges ($[-20, 85]^\circ\text{C}$ for temperature, $[2, 400]\text{cm}$ for ultrasonic).

---

## 7. Two-Tier Hybrid Diagnosis & Adjudication Arbiter

The hybrid engine ([`diagnosis_loc/hybrid_diagnosis.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/diagnosis_loc/hybrid_diagnosis.py)) fuses statistical pattern classification with formal mathematical proofs:

### 7.1 Adjudication Matrix

| Formal TFSM Verdict | Statistical DL Verdict | Hybrid Outcome | Description & Operational Response |
| :---: | :---: | :--- | :--- |
| **Violation** | **Attack** | `CONSENSUS` | Full consensus. Both tiers confirm an ongoing attack. Output includes DL attack category and formal mutant localization. |
| **Violation** | **Normal** | `FORMAL_VIOLATION_DL_NORMAL` | High-priority safety alert. Formal state machine detected a mathematical timing/protocol breach that statistical DL missed. |
| **Pass** | **Attack** | `STATISTICAL_ANOMALY` | Early-warning behavioral anomaly. Statistical DL detected an anomalous temporal pattern while formal constraints are still nominally satisfied. |
| **Pass** | **Normal** | `NOMINAL` | Benign operation verified by both statistical and formal tiers. |
| — | — | `INSUFFICIENT_SEQUENCE` | Fewer than 16 events in session; DL windowing cannot engage; falls back to formal diagnosis. |
| — | — | `MODEL_UNAVAILABLE` | DL checkpoint unreadable or disabled; formal TFSM operates autonomously. |

> [!NOTE]
> **Separation of Evidence Guarantee:** Softmax confidence scores are reported strictly as empirical neural model scores, not Bayesian probabilities. Formal evidence (transition IDs, observed/expected symbols, mutant candidates) is never synthesized from DL predictions, and DL predictions are never converted directly into formal TFSM proofs.

---

## 8. Verification & Test Suite Inventory

The system includes **125 unit tests across 13 test suites**, achieving 100% pass rate:

| Test Suite File | Test Count | Scope / Component Tested | Status |
| :--- | :---: | :--- | :---: |
| [`tests/test_fsm_sensors.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_fsm_sensors.py) | 21 | Four sensor TFSM models, state transitions, timing constraints | **PASS** |
| [`tests/test_unified_simulator.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_unified_simulator.py) | 8 | Multi-sensor discrete-event simulator, clock advances, isolated runtimes | **PASS** |
| [`tests/test_attack_injector.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_attack_injector.py) | 8 | Six attack scenarios (DoS, Replay, MITM, DataFalsification, Delay, IdentitySpoofing) | **PASS** |
| [`tests/test_phase3_data.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_phase3_data.py) | 10 | Real MQTT dataset parsing, packet schema validation | **PASS** |
| [`tests/test_stage3_dataset.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_stage3_dataset.py) | 9 | Four-sensor generated dataset generation, session integrity, manifest validation | **PASS** |
| [`tests/test_stage3b_baselines.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_stage3b_baselines.py) | 3 | Real-data tabular ML baselines (RandomForest, XGBoost) | **PASS** |
| [`tests/test_stage3b6_improvements.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_stage3b6_improvements.py) | 3 | Real-data feature engineering, calibration metrics | **PASS** |
| [`tests/test_stage3b6_runtime.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_stage3b6_runtime.py) | 5 | Real-data inference latency benchmarks and memory bounds | **PASS** |
| [`tests/test_stage3c_sequences.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_stage3c_sequences.py) | 8 | Sequence windowing, batch collation, LSTM/BiLSTM forward passes | **PASS** |
| [`tests/test_unified_diagnosis.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_unified_diagnosis.py) | 16 | Formal TFSM diagnosis engine, sensor isolation, mutant localization | **PASS** |
| [`tests/test_stage5_dataset.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_stage5_dataset.py) | 14 | 14 observable features, zero leakage audit, MaskedStandardScaler, target schema mapping | **PASS** |
| [`tests/test_stage5_training.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_stage5_training.py) | 8 | Multi-seed training engine, early stopping, validation Macro-F1 checkpoint selection | **PASS** |
| [`tests/test_stage5_hybrid.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_stage5_hybrid.py) | 15 | Two-tier hybrid engine, adjudication arbiter, fallback modes, evidence isolation, dataclass ordering | **PASS** |
| [`tests/test_stage6_integration.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_stage6_integration.py) | 16 | Stage 6 end-to-end integration, scenario validation, outcome states | **PASS** |
| **TOTAL** | **144** | **Complete Full Regression Suite** | **144/144 PASS** |

---

## 9. Windows CMD Execution Instructions

Follow these exact commands in a Windows Command Prompt (`cmd.exe`) to execute any stage of the system:

### 9.1 Environment Setup
```cmd
cd /d c:\Users\cheru\Downloads\iot-ids-diagnosis\iot-ids-diagnosis
call .venv_stage3c\Scripts\activate
```

### 9.2 Formal Mutation Verification (Stage 1)
```cmd
python -m models_fsm.verify
```
*Expected Output:* `ALL 59 TESTS KILL THEIR MUTANTS (59/59)` with exit code 0.

### 9.3 End-to-End System Integration Harness (Stages 1–6)
```cmd
python run_full_system.py
```
*Expected Output:* Executes formal verification, runs multi-sensor benign and DoS attack simulations, executes two-tier hybrid diagnosis, verifies zero leakage, runs all 125 regression unit tests, and outputs `STATUS: ALL CHECKS PASSED`.

### 9.4 Full Unit Test Suite Execution
```cmd
python -m unittest discover -s tests -p "test_*.py" -v
```
*Expected Output:* `Ran 125 tests in ~5s ... OK`.

### 9.5 Individual Test Suites
```cmd
:: Formal FSM sensor unit tests
python -m unittest tests.test_fsm_sensors -v

:: Unified simulator and attack injector tests
python -m unittest tests.test_unified_simulator -v
python -m unittest tests.test_attack_injector -v

:: Formal diagnosis engine tests
python -m unittest tests.test_unified_diagnosis -v

:: Stage 5 dataset pipeline tests
python -m unittest tests.test_stage5_dataset -v

:: Stage 5 training engine tests
python -m unittest tests.test_stage5_training -v

:: Stage 5.3 hybrid diagnosis tests
python -m unittest tests.test_stage5_hybrid -v
```

### 9.6 Multi-Seed Sequence Model Training (Optional Full Retraining)
```cmd
python -m ids.train_four_sensors_sequences --model all --epochs 50 --device cpu
```

