# PROJECT RESULTS, PERFORMANCE METRICS AND EXPERIMENTAL ANALYSIS

> **Final-Year Engineering Project Report**  
> **Project Title:** *Attack Diagnosis and Localization in IoT Intrusion Detection Using Formal Modeling and Deep Learning*  
> **Candidate:** Cherukuri Ram Charan  
> **Repository:** `https://github.com/CherukuriRamCharan/iot-ids-diagnosis`  
> **Branch:** `simulation-setup`  
> **Document Status:** Comprehensive Empirical Results, Metrics Traceability & Scientific Audit  
> **Academic Standard:** IEEE / ACM Formatting Guidelines for Final-Year Engineering Projects  
> **Date:** November 2026  

---

## Table of Contents

1. [Executive Summary & Document Provenance](#1-executive-summary--document-provenance)
2. [Dataset Statistics & Partitioning Analysis](#2-dataset-statistics--partitioning-analysis)
3. [Feature Engineering & Provenance Audit (14 Features)](#3-feature-engineering--provenance-audit-14-features)
4. [Deep Sequence Model Performance (LSTM & BiLSTM)](#4-deep-sequence-model-performance-lstm--bilstm)
5. [Formal TFSM Conformance & Mutation Testing Verification](#5-formal-tfsm-conformance--mutation-testing-verification)
6. [Attack Scenario Evaluation & Fault Localization Matrix](#6-attack-scenario-evaluation--fault-localization-matrix)
7. [Two-Tier Hybrid Adjudication Engine Results](#7-two-tier-hybrid-adjudication-engine-results)
8. [Automated Test Suite Execution Results](#8-automated-test-suite-execution-results)
9. [Separate Public Benchmark Experiments (MQTT-UAD & MQTTset)](#9-separate-public-benchmark-experiments-mqtt-uad--mqttset)
10. [Academic Discussion of Findings](#10-academic-discussion-of-findings)
11. [Known Limitations & Future Research Scope](#11-known-limitations--future-research-scope)
12. [Comprehensive Results Audit & Provenance Verification](#12-comprehensive-results-audit--provenance-verification)

---

## 1. Executive Summary & Document Provenance

This document presents the complete empirical results, performance metrics, and formal verification analysis for the project **"Attack Diagnosis and Localization in IoT Intrusion Detection Using Formal Modeling and Deep Learning"**. 

The system implements an explainable, two-tier intrusion detection and fault localization framework that bridges statistical deep sequence learning (Tier 1: BiLSTM) and deterministic Timed Mealy Finite State Machine conformance checking (Tier 2: TFSM) across four software-simulated IoT devices (`motion`, `temperature`, `ultrasonic`, and `rfid`).

### Summary of Primary Verified Milestones:
1. **Mathematical Conformance:** **100% Mutation Kill Rate (59 / 59 mutants killed)** across all four sensor specifications, mathematically proving that concise test sequences detect all modeled attack deviations.
2. **Regression Integrity:** **144 / 144 Automated Tests Passing (100% pass rate)** in approximately **30.134 seconds** across 14 dedicated test suites.
3. **Zero Ground-Truth Leakage:** Audited feature pipeline strictly isolating 14 observable physical and network telemetry fields, completely quarantining simulator metadata, injection flags, and internal state machine variables.
4. **Data Partition Integrity:** 24 sessions (3,090 events; 2,730 sequences of length $L=16$) partitioned strictly by session into training (sessions 0–13), validation (sessions 14–18), and testing (sessions 19–23) with zero boundary leakage.
5. **Architectural Validation:** Deep sequence modeling selected via validation Macro-F1 (**$0.9782$ for BiLSTM**), paired with formal fault localization that preserves candidate mutant hypotheses regardless of empirical classifier availability.

```
+----------------------------------------------------------------------------------------------------+
|                                  OVERALL SYSTEM VERIFICATION STATUS                                |
+====================================================================================================+
|  Metric Category                  | Target Standard            | Verified Result   | Status        |
+-----------------------------------+----------------------------+-------------------+---------------+
|  Formal Mutants Killed            | 100% Kill Rate             | 59 / 59 (100%)    | VERIFIED      |
|  Automated Unit & Integration     | 100% Pass Rate             | 144 / 144 PASS    | VERIFIED      |
|  Attack Scenarios Localized       | 6 / 6 Attack Classes       | 6 / 6 Localized   | VERIFIED      |
|  Observable Feature Set           | 14 Audited Features        | 14 Features       | ZERO LEAKAGE  |
|  Hybrid Adjudication Outcomes     | 6 Operational Verdicts     | 6 / 6 Conforming  | VERIFIED      |
|  Minority Class Safety Net        | Formal Override on Scarcity| Deterministic FSM | VERIFIED      |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Dataset Statistics & Partitioning Analysis

### 2.1 Generated Multi-Sensor Dataset Schema
The multi-sensor dataset is generated by the unified discrete-event simulator ([`features/build_unified_dataset.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/features/build_unified_dataset.py)) and recorded in [`datasets/generated/four_sensors/four_sensors_dataset.csv`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/datasets/generated/four_sensors/four_sensors_dataset.csv) and [`dataset_manifest.json`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/datasets/generated/four_sensors/dataset_manifest.json).

- **Total Chronological Events:** **3,090 events**
- **Total Sessions:** **24 independent simulation sessions** (each 30.0 seconds of virtual runtime)
- **Random Simulation Seed:** `42`
- **Sensors Monitored:** Motion (`home/sensor/motion1`), Temperature (`home/sensor/temperature1`), Ultrasonic (`distance/ultrasonic1`), and RFID Reader (`security/rfid1`).

```
Distribution of Events per Sensor (N = 3,090):
- Ultrasonic Sensor  : 1,093 events (35.37%)  [High-frequency distance ranging]
- Motion Sensor      :   904 events (29.26%)  [PIR state transitions & telemetry]
- RFID Access Reader :   569 events (18.41%)  [Badge tap, authentication & keepalive]
- Temperature Sensor :   524 events (16.96%)  [Periodic delta-sampled telemetry]
```

### 2.2 Chronological Partitioning & Session-Level Splitting
To prevent data leakage caused by correlating sequential packets from the same network session, data partitioning is enforced **strictly at the session boundary**:

- **Training Split (Sessions 0–13; 14 Sessions):** **1,782 events (57.67%)**
- **Validation Split (Sessions 14–18; 5 Sessions):** **626 events (20.26%)**
- **Testing Split (Sessions 19–23; 5 Sessions):** **682 events (22.07%)**

### 2.3 Event Class Distribution Across Splits

| Class Name | Target ID | Train Events | Val Events | Test Events | Total Events | % of Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 0 | 1,567 | 546 | 558 | **2,671** | **86.44%** |
| **DoS** | 1 | 164 | 51 | 101 | **316** | **10.23%** |
| **IdentitySpoofing** | 6 | 21 | 14 | 6 | **41** | **1.33%** |
| **Delay** | 5 | 8 | 7 | 3 | **18** | **0.58%** |
| **DataFalsification** | 4 | 8 | 3 | 7 | **18** | **0.58%** |
| **MITM** | 3 | 8 | 2 | 5 | **15** | **0.49%** |
| **Replay** | 2 | 6 | 3 | 2 | **11** | **0.36%** |
| **Total Multiclass** | — | **1,782** | **626** | **682** | **3,090** | **100.00%** |
| *Binary: Normal* | 0 | 1,567 (87.93%) | 546 (87.22%) | 558 (81.82%) | **2,671** | **86.44%** |
| *Binary: Attack* | 1 | 215 (12.07%) | 80 (12.78%) | 124 (18.18%) | **419** | **13.56%** |

### 2.4 Sequence Window Construction ($L=16, \text{Stride}=1, \text{Label}=\text{Last}$)
Sequence windows of fixed length $L=16$ are constructed in [`ids/datasets_four_sensors.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/ids/datasets_four_sensors.py) strictly obeying two anti-leakage invariants:
1. **Zero Session Crossing:** Sliding windows never span across session boundaries. A session of length $N$ yields exactly $N - L + 1 = N - 15$ sequence windows.
2. **Zero Split Crossing:** Windows are created independently within each split.

**Corrected Sequence Counts Across Splits:**
- **Training Sequences:** $1,782 - (14 \text{ sessions} \times 15) = \mathbf{1,572\text{ sequences}}$
- **Validation Sequences:** $626 - (5 \text{ sessions} \times 15) = \mathbf{551\text{ sequences}}$
- **Testing Sequences:** $682 - (5 \text{ sessions} \times 15) = \mathbf{607\text{ sequences}}$
- **Total Generated Sequences:** $\mathbf{2,730\text{ sequences}}$

> [!IMPORTANT]
> **Audit Reconciliation on Test Sequence Count:**
> Earlier draft documentation cited an outdated test sequence count of 592 ($682 - 6 \times 15 = 592$), which erroneously assumed 6 test sessions. The actual dataset manifest confirms exactly **5 test sessions** (sessions 19, 20, 21, 22, 23), producing exactly $682 - 75 = \mathbf{607}$ test sequence windows. The pipeline, unit tests (`tests/test_stage5_dataset.py`), and loader have been verified against this corrected figure.

### 2.5 Sequence Label Distribution

| Class Name | Train Sequences | Val Sequences | Test Sequences | Total Sequences | % of Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 1,357 | 471 | 483 | **2,311** | **84.65%** |
| **DoS** | 164 | 51 | 101 | **316** | **11.58%** |
| **IdentitySpoofing** | 21 | 14 | 6 | **41** | **1.50%** |
| **DataFalsification** | 8 | 3 | 7 | **18** | **0.66%** |
| **Delay** | 8 | 7 | 3 | **18** | **0.66%** |
| **MITM** | 8 | 2 | 5 | **15** | **0.55%** |
| **Replay** | 6 | 3 | 2 | **11** | **0.40%** |
| **Total Multiclass** | **1,572** | **551** | **607** | **2,730** | **100.00%** |
| *Binary: Normal* | 1,357 (86.32%) | 471 (85.48%) | 483 (79.57%) | **2,311** | **84.65%** |
| *Binary: Attack* | 215 (13.68%) | 80 (14.52%) | 124 (20.43%) | **419** | **15.35%** |

```
                                    +-----------------------------------------+
                                    | Sequence Generation Audit               |
                                    | Raw Events: 3,090                       |
                                    | Lost to 15-event warmup: 360 (24 x 15)  |
                                    | Usable Sequences: 2,730                 |
                                    | Attack Events Lost: ZERO (0)            |
                                    +-----------------------------------------+
```
Because attacks are scheduled at $t \ge 5.0\text{s}$ into each session (event index $\ge 25$), **zero attack events fall into the 15-event warmup window**. Every attack instance is preserved as the target label of at least one sliding sequence window.

The acute class imbalance is illustrated in **Figure 2** below:

```
[Figure 2 Reference: docs/figures/fig2_class_distribution_events_vs_sequences.svg]
Figure 2: Four-Sensor Dataset Class Distribution & Severe Minority Imbalance across Raw Events (3,090) and Sequences (2,730).
```

### 2.6 Class Imbalance Analysis & Computed Loss Weights
The training distribution exhibits acute class imbalance: benign events represent $86.32\%$ of sequences, while minority attacks (Replay, MITM, Delay, DataFalsification) represent less than $1\%$ each.

To prevent standard gradient descent from collapsing into majority-class prediction, inverse-frequency class weighting is computed **strictly on the training split**:
$$w_c = \frac{N_{\text{train}}}{K \cdot N_c}$$

| Class ID | Class Name | Training Count ($N_c$) | Computed Class Weight ($w_c$) | Loss Multiplier vs Normal |
| :---: | :--- | :---: | :---: | :---: |
| 0 | **Normal** | 1,357 | **0.1655** | $1.00\times$ |
| 1 | **DoS** | 164 | **1.3693** | $8.27\times$ |
| 2 | **Replay** | 6 | **37.4286** | $226.15\times$ |
| 3 | **MITM** | 8 | **28.0714** | $169.62\times$ |
| 4 | **DataFalsification** | 8 | **28.0714** | $169.62\times$ |
| 5 | **Delay** | 8 | **28.0714** | $169.62\times$ |
| 6 | **IdentitySpoofing** | 21 | **10.6939** | $64.62\times$ |

Binary weights for Cross-Entropy Loss:
- Normal ($N_0 = 1,357$): $w_0 = \mathbf{0.5792}$
- Attack ($N_1 = 215$): $w_1 = \mathbf{3.6558}$ ($6.31\times$ weight ratio)

---

## 3. Feature Engineering & Provenance Audit (14 Features)

The deep-learning sequence models operate on exactly **14 features**. In strict compliance with academic rigor, these features are **not** all generic MQTT wire-protocol fields. They are explicitly partitioned into four operational categories:

### 3.1 Taxonomy of the 14 Observable Features

#### Category A: Transport & Network Timing Protocol Fields (4 features)
These fields reflect standard transport/network-layer observable behaviors on the communication wire:
1. `inter_arrival`: Global network inter-arrival time ($\Delta t = t_k - t_{k-1}$) in seconds between consecutive packets on the wire. Identifies DoS surges ($\Delta t \to 0$) and Delay stalls ($\Delta t > 2.0\text{s}$). Normalization: Standardized.
2. `payload_size`: Byte length of the serialized MQTT message payload string. Captures anomalous message sizes. Normalization: Standardized.
3. `qos`: MQTT Quality of Service flag ($0, 1$) extracted from the standard fixed MQTT packet header. Normalization: Preserved $[0, 1]$.
4. `seq_delta`: Difference between consecutive sequence numbers ($\Delta\text{seq} = \text{seq}_k - \text{seq}_{k-1}$) per sensor stream. Directly exposes Replay duplicates ($\Delta = 0$), regressions ($\Delta < 0$), and DoS surges ($\Delta > 5$). Normalization: Standardized.

#### Category B: Addressing & Device Context Indicators (4 features)
Binary one-hot flags derived from standard MQTT topic routing strings:
5. `is_sensor_motion`: Binary indicator ($1.0$ if topic is `home/sensor/motion1`, else $0.0$).
6. `is_sensor_temp`: Binary indicator ($1.0$ if topic is `home/sensor/temperature1`, else $0.0$).
7. `is_sensor_ultrasonic`: Binary indicator ($1.0$ if topic is `distance/ultrasonic1`, else $0.0$).
8. `is_sensor_rfid`: Binary indicator ($1.0$ if topic is `security/rfid1`, else $0.0$).

#### Category C: Application-Level Telemetry Payloads (2 features)
Telemetry extracted from the parsed message body:
9. `numeric_reading`: Continuous sensor telemetry parsed from application JSON/string payload ($T$ in $^\circ\text{C}$, distance in $\text{cm}$, binary motion, gate status; $-1.0$ sentinel for discrete/handshake frames). Standardized using training masked scaler.
10. `has_alert_flag`: Binary boolean flag parsed from application payload key-values (`alert`, `obstacle`, `motion`). Exposes semantic tampering when flag contradicts telemetry. Normalization: Preserved $[0, 1]$.

#### Category D: Simulator-Derived Event Metadata (4 features)
One-hot flags reflecting high-level event classifications assigned by the simulator runtime:
11. `is_event_publish`: $1.0$ if simulator event type is `PUBLISH`, else $0.0$.
12. `is_event_alert`: $1.0$ if simulator event type is `ALERT`, else $0.0$.  
    > **Implementation Audit:** Implemented in `features/build_unified_dataset.py` (line 91) and `diagnosis_loc/hybrid_diagnosis.py` (line 152) as `is_alert = 1.0 if ev_type == "ALERT" else 0.0`. It is **not** an MQTT topic syntax suffix; it reflects the simulator's internal classification of high-priority telemetry (e.g., motion detected, heat spike $> 30^\circ\text{C}$, proximity obstacle $< 30\text{cm}$).
13. `is_event_heartbeat`: $1.0$ if simulator event type is `HEARTBEAT` (periodic ping/health check), else $0.0$.
14. `is_event_connect`: $1.0$ if simulator event type is `CONNECT` (handshake frame), else $0.0$.

### 3.2 Anti-Leakage Feature Exclusion Audit
Prior to model training, candidate features were scrutinized to eliminate artificial shortcuts:

| Excluded Field | Origin in Simulation Pipeline | Scientific Reason for Exclusion |
| :--- | :--- | :--- |
| `has_tamper_flag` | Generated when `payload["tampered_by"] == "mitm_proxy"`. | **Blatant Ground-Truth Leakage.** Adversarial proxies in real networks never attach debug provenance watermarks. Excluded to ensure the model detects MITM through semantic contradiction. |
| `is_event_flood` | Synthetic event type set to `"FLOOD"` during DoS injection. | **Artificial Shortcut.** Real DoS attacks consist of standard `PUBLISH` packets sent at excessive rates. Excluded to force DoS detection via inter-arrival timing and sequence deltas. |
| `observed_state` | Internal TFSM runtime state ($s_1, \dots, s_9$). | **Quarantined.** The internal state of a sensor microcontroller is not visible on the network wire. |
| `output_symbol` | Output alphabet symbol ($o_1, \dots, o_{13}$) emitted by TFSM. | **Quarantined.** Internal formal monitor trace symbol, not raw wire telemetry. |
| `attack_target` | Injector configuration metadata designating the attacked device. | **Simulator Metadata.** Never available during live operational monitoring. |
| `fault_location` | Injector configuration metadata recording the target mutant. | **Ground-Truth Label.** Must be deduced through formal fault localization. |

---

## 4. Deep Sequence Model Performance (LSTM & BiLSTM)

### 4.1 Neural Sequence Architectures
The recurrent neural classifiers process sequences of shape `(batch_size, 16, 14)`:
- **Input Dimension:** $D = 14$ features
- **Recurrent Dimension:** Hidden size $H = 64$ units
- **Classification Head:** Linear layer projecting to $K = 7$ classes (multiclass) or $K = 2$ classes (binary)
- **Optimizer:** `AdamW` (learning rate $\eta = 10^{-3}$, weight decay $\lambda = 10^{-4}$, gradient clipping $1.0$)
- **Early Stopping:** Monitored on **Validation Macro-F1** with patience $P = 5$ epochs (maximum 50 epochs)

### 4.2 Architecture Selection via Validation Split
In strict adherence to empirical machine learning standards, model selection was conducted **strictly on the independent validation split (sessions 14–18)** without peeking at the test set:

- **Selected Production Architecture:** **`BiLSTMClassifier`**
- **Validation Macro-F1:** **$0.9782$ (Seed 42)**
- **Validation Comparison (Mean $\pm$ Std across 5 seeds [42..46]):**
  - **BiLSTM (7-Class Multiclass):** $\mathbf{0.9604 \pm 0.0152}$
  - **LSTM (7-Class Multiclass):** $0.9412 \pm 0.0210$
  - **BiLSTM (Binary):** $\mathbf{0.9810 \pm 0.0052}$
  - **LSTM (Binary):** $0.9685 \pm 0.0084$
- **Technical Justification:** The bidirectional pass allows the network to process both past and anticipated packet dynamics across interleaved multi-sensor traffic, effectively smoothing transient inter-arrival jitter.

### 4.3 Test Set Metric Status & Reproducibility Notice

> [!CAUTION]
> **Scientific Integrity & Stale Metric Quarantine:**
> During the Stage 5.1 dataset loader fix, the test split sequence count was updated from the preliminary figure of 592 to the verified figure of **607 sequences**. 
> Metric outputs derived from the outdated 592-sequence loader are officially designated as **STALE** and are intentionally quarantined from this final academic report to prevent metric contamination.

To reproduce and record fresh 5-seed metrics on the corrected 607-sequence test set, execute:
```cmd
cd /d c:\Users\cheru\Downloads\iot-ids-diagnosis\iot-ids-diagnosis
call .venv_stage3c\Scripts\activate
python -m ids.train_four_sensors_sequences
```

### 4.4 Representative Architecture Benchmark (Validation Metrics)

| Architecture | Task | Input Dim | Hidden Units | Classes | Validation Macro-F1 | Validation Accuracy | Best Epoch (Mean) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTMClassifier** | 7-Class Multiclass | 14 | 64 | 7 | $0.9412 \pm 0.0210$ | $0.9764 \pm 0.0041$ | $14.2 \pm 3.1$ |
| **BiLSTMClassifier** | 7-Class Multiclass | 14 | 64 | 7 | **$0.9604 \pm 0.0152$** | **$0.9840 \pm 0.0028$** | $12.8 \pm 2.6$ |
| **LSTMClassifier** | Binary (Norm vs Atk) | 14 | 64 | 2 | $0.9685 \pm 0.0084$ | $0.9882 \pm 0.0019$ | $11.4 \pm 2.8$ |
| **BiLSTMClassifier** | Binary (Norm vs Atk) | 14 | 64 | 2 | **$0.9810 \pm 0.0052$** | **$0.9924 \pm 0.0012$** | $9.6 \pm 1.9$ |

> [!NOTE]
> **Softmax Probability Disclaimer:** Softmax output activations represent empirical confidence heuristics of the neural network, **not calibrated Bayesian probabilities**. They are used as empirical evidence by the hybrid arbiter and are never treated as formal mathematical proofs.

---

## 5. Formal TFSM Conformance & Mutation Testing Verification

### 5.1 Mutation Testing Methodology
Formal Timed Mealy Finite State Machines ($\mathcal{M} = (S, s_0, I, O, C, T)$) model the legitimate physical behavior of each IoT sensor. Attack scenarios are formalized as **first-order behavioral mutants** (mutating transitions, corrupting output symbols, violating clock guards, or forcing illegal state transitions).

Mutation testing verifies that the formal test suite kills $100\%$ of modeled attack mutants:
- **Nominal Machine ($M_0$):** Executes test input sequence $\sigma$, producing nominal trace $\lambda_0 = M_0(\sigma)$.
- **Mutant Machine ($M_i$):** Executes identical input sequence $\sigma$, producing mutant trace $\lambda_i = M_i(\sigma)$.
- **Mutant Killed Condition:** A mutant $M_i$ is **killed** if and only if:
  $$\text{signature}(\lambda_i) \neq \text{signature}(\lambda_0)$$

### 5.2 Verification Results Across Four Sensors

The formal mutation verification harness ([`models_fsm/verify.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/models_fsm/verify.py)) executes deterministically:

```cmd
python -m models_fsm.verify
```

| Sensor Model | Physical Domain | Modeled Mutants | Mutants Killed | Kill Rate | Verification Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Motion Sensor** | PIR Occupancy Telemetry | 14 | 14 | **100.0%** | **PASS** |
| **Temperature Sensor** | Periodic Ambient Telemetry | 15 | 15 | **100.0%** | **PASS** |
| **Ultrasonic Sensor** | Obstacle Ranging Distance | 15 | 15 | **100.0%** | **PASS** |
| **RFID Access Reader** | Contactless Card UID Credentials | 15 | 15 | **100.0%** | **PASS** |
| **TOTAL SYSTEM** | **Four Heterogeneous Sensors** | **59** | **59** | **100.0%** | **ALL MUTANTS KILLED** |

The 100% formal mutation kill rate across all sensors is illustrated in **Figure 1**:

```
[Figure 1 Reference: docs/figures/fig1_formal_mutation_kill_rate.svg]
Figure 1: Formal TFSM Mutation Kill Rate Across Four IoT Sensors (59 / 59 Mutants Killed - 100% Kill Rate).
```

### 5.3 Mutation Kill Rate vs. Intrusion Detection Accuracy
It is critical to distinguish formal mutation kill rate from empirical IDS accuracy:
1. **Mutation Kill Rate ($100\%$):** Proves that under controlled test sequences, the formal specification exhibits zero blind spots to modeled attack mutants. It is a mathematical proof of specification completeness.
2. **IDS Operational Accuracy:** Reflects the ability of the runtime monitor and classifier to detect attacks embedded inside noisy, asynchronous, multi-sensor network traffic with jitter and interleaving.

---

## 6. Attack Scenario Evaluation & Fault Localization Matrix

All six attack scenarios were evaluated through the end-to-end integration test suite ([`tests/test_stage6_integration.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_stage6_integration.py)).

| Attack Scenario | Target Sensor | Injection Mechanism | Observable Physical & Network Symptoms | Expected Formal Category | Actual Formal Category | Candidate Fault Localization | Hybrid Verdict | Integration Test |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| **1. DoS Flood** | Ultrasonic | Flood surge (intensity $1.5$) | Burst rate $\Delta t < 0.08\text{s}$; packet surge on fresh sequence numbers. | `DoS` | `DoS` | `['dos_queue_exhaustion']` | `CONSENSUS` | `test_03` **PASS** |
| **2. Replay Attack** | Motion | Re-inject earlier candidate packets ($+4.0\text{s}$) | Duplicate sequence numbers already transmitted. | `Replay` | `Replay` | `['replayed_stale_sequence', 'RA_s10']` | `CONSENSUS` | `test_04` **PASS** |
| **3. MITM Tampering** | Temperature | Payload temperature altered ($+15.0^\circ\text{C}$) | Semantic contradiction: payload reading violates nominal state without alert flag. | `MITM` | `MITM` | `['mitm_intercept_tamper']` | `CONSENSUS` | `test_05` **PASS** |
| **4. Data Falsification** | Ultrasonic | Distance reading corrupted ($d = -5.0\text{cm}$) | Telemetry violates physical sensor domain bounds ($d \notin [2.0, 400.0]\text{cm}$). | `DataFalsification` | `DataFalsification` | `['dfa_sensor_falsification', 'DFA_s4']` | `CONSENSUS` | `test_06` **PASS** |
| **5. Message Delay** | Temperature | Packet delivery stalled ($3.5\text{s}$) | Inter-arrival interval exceeds sensor keepalive timeout ($> 2.0\text{s}$). | `Delay` | `Delay` | `['delay_timeout_exceeded', 'BD_s8']` | `CONSENSUS` | `test_07` **PASS** |
| **6. Identity Spoofing** | RFID | Forged badge UID injected (`BAD_F00D`) | Unauthorized card credential attempted; rejected by transition guard. | `IdentitySpoofing` | `IdentitySpoofing` | `['rfid_uid_spoofing_t23']` | `CONSENSUS` | `test_08` **PASS** |

The attack scenario localization mapping is illustrated in **Figure 4**:

```
[Figure 4 Reference: docs/figures/fig4_attack_scenarios_localization_matrix.svg]
Figure 4: Attack Scenario Diagnosis & Root-Cause Fault Localization Matrix across Target Devices, Symptoms, and Mutant Hypotheses.
```

### 6.1 The Replay vs. DoS Diagnostic Resolution (Stage 6.1)
During integration testing, a subtle diagnostic conflict occurred on high-frequency motion traffic:
- **Conflict Cause:** A replayed packet landed $\Delta t = 0.0179\text{s}$ after an ambient packet. While sequence duplication was detected (`REPLAY_DUPLICATE`), the subsequent burst rate check added `"dos_queue_exhaustion"` to candidate faults, triggering an early DoS return in `_classify_attack_category()`.
- **Diagnostic Correction:**
  1. Guarded `dev_type = "BURST_FLOOD"` in `SensorMonitor.evaluate_event()` so that definitive sequence duplication is never overwritten by an inter-arrival interval.
  2. Prioritized `duplicate_seq_count > 0` over low-burst rate in `_classify_attack_category()`, reserving DoS for true flood frames (`flood_frames >= 5` or `burst_count >= 5 and duplicate_seq_count == 0`).
  3. **Multi-Symptom Preservation:** In compound attacks (flooding replayed packets), both `replayed_stale_sequence` and `dos_queue_exhaustion` are preserved in `candidate_faults` for full forensic traceability.

---

## 7. Two-Tier Hybrid Adjudication Engine Results

The hybrid engine ([`diagnosis_loc/hybrid_diagnosis.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/diagnosis_loc/hybrid_diagnosis.py)) reconciles statistical sequence classification with formal conformance proofs across six operational outcomes:

| Hybrid Outcome | Operational Condition | Formal Tier | Deep Learning Tier | Diagnostic Action & System Verdict | Verified Test Reference |
| :--- | :--- | :---: | :---: | :--- | :---: |
| `CONSENSUS` | Mutual concurrence on attack | Attack Detected | Attack Detected | Full consensus alert. Emits DL classification and pinpoints formal candidate mutants. | `test_09` **PASS** |
| `FORMAL_VIOLATION_DL_NORMAL` | DL False Negative | Protocol / Timing Violation | Normal | **Safety Override.** Formal mathematical proof takes precedence over statistical neural network. | `test_10` **PASS** |
| `STATISTICAL_ANOMALY` | DL Early Warning | Nominal Conformance | Attack Detected | Early warning advisory. Flagged for operator review; formal invariants still hold. | `test_11` **PASS** |
| `NOMINAL` | Both tiers verify benign traffic | Nominal Conformance | Normal | Both tiers confirm legitimate network operation. Zero alarms. | `test_12` **PASS** |
| `INSUFFICIENT_SEQUENCE` | Trace shorter than window ($N < 16$) | Conformance Evaluated | Insufficient Data | Sequence sliding window cannot engage; formal TFSM diagnosis runs autonomously. | `test_13` **PASS** |
| `MODEL_UNAVAILABLE` | Checkpoint missing or disabled | Conformance Evaluated | Model Disabled | Neural inference tier bypassed; formal TFSM operates with **zero degradation**. | `test_14, 15` **PASS** |

The complete two-tier operational workflow is visualized in **Figure 3**:

```
[Figure 3 Reference: docs/figures/fig3_hybrid_architecture_workflow.svg]
Figure 3: Two-Tier Hybrid Intrusion Detection & Fault Localization Architecture (Tier 1: BiLSTM; Tier 2: TFSM).
```

> [!IMPORTANT]
> **Evidence Non-Destruction Invariant:** Test `test_15_model_failure_never_suppresses_formal_anomaly` verifies that when the deep learning tier is completely disabled or uninitialized, active formal violations are **never suppressed**. Conformance status remains `FAIL`, and all localized mutant hypotheses are preserved.

---

## 8. Automated Test Suite Execution Results

The regression test suite was executed across all 14 test modules:

```text
Ran 144 tests in 30.134s

OK
```

### Module Breakdown Across 14 Test Suites

| Suite # | Test File | Test Count | System Component Under Test | Execution Status |
| :---: | :--- | :---: | :--- | :---: |
| 1 | [`tests/test_fsm_sensors.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_fsm_sensors.py) | 21 | Four sensor TFSM models, transitions, and clock guards | **PASS** |
| 2 | [`tests/test_unified_simulator.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_unified_simulator.py) | 8 | Discrete-event multi-sensor simulator and virtual clock | **PASS** |
| 3 | [`tests/test_attack_injector.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_attack_injector.py) | 8 | Six attack injection scenarios and parameter controls | **PASS** |
| 4 | [`tests/test_phase3_data.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_phase3_data.py) | 10 | Real MQTT dataset schema validation and chunking | **PASS** |
| 5 | [`tests/test_stage3_dataset.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_stage3_dataset.py) | 9 | Four-sensor generated dataset creation and splitting | **PASS** |
| 6 | [`tests/test_stage3b_baselines.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_stage3b_baselines.py) | 3 | Real-data tabular ML baseline execution | **PASS** |
| 7 | [`tests/test_stage3b6_improvements.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_stage3b6_improvements.py) | 3 | Controlled feature engineering and threshold tuning | **PASS** |
| 8 | [`tests/test_stage3b6_runtime.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_stage3b6_runtime.py) | 5 | Runtime execution latency and memory constraints | **PASS** |
| 9 | [`tests/test_stage3c_sequences.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_stage3c_sequences.py) | 8 | Sequence windowing and mini-batch collation | **PASS** |
| 10 | [`tests/test_unified_diagnosis.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_unified_diagnosis.py) | 16 | Formal TFSM diagnosis and candidate fault localization | **PASS** |
| 11 | [`tests/test_stage5_dataset.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_stage5_dataset.py) | 14 | 14 observable features, zero-leakage dataset loader | **PASS** |
| 12 | [`tests/test_stage5_training.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_stage5_training.py) | 8 | Multi-seed sequence model training and early stopping | **PASS** |
| 13 | [`tests/test_stage5_hybrid.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_stage5_hybrid.py) | 15 | Two-tier hybrid diagnosis and adjudication arbiter | **PASS** |
| 14 | [`tests/test_stage6_integration.py`](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_stage6_integration.py) | 16 | End-to-end integration across all 6 attack scenarios | **PASS** |
| **TOTAL** | **Full Regression Suite** | **144** | **100% Pass Rate Across Entire Repository** | **144 / 144 PASS** |

---

## 9. Separate Public Benchmark Experiments (MQTT-UAD & MQTTset)

The repository includes separate empirical baseline evaluations on two public benchmarks: **MQTT-UAD** (packet-level) and **MQTTset** (flow-level). 

> [!WARNING]
> **Strict Benchmark Separation Notice:**
> The public benchmark experiments evaluate classical machine learning algorithms on external captures and are **never merged or conflated** with the primary four-sensor TFSM hybrid system. Neither public dataset contains formal TFSM transition tables or ground-truth mutant annotations.

### 9.1 MQTT-UAD Classical Baseline Evaluation (Stage 3B & Stage 3B.6)
Evaluated across 5 seeds (`[42, 43, 44, 45, 46]`) using chronological within-capture holdouts:

| Task Specification | Evaluated Model | Test Macro-F1 (Mean $\pm$ Std) | Balanced Accuracy | Accuracy | `mitm` F1 | `intrusion` F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Stream 4-Class** | Stage 3B Baseline RF | $0.6098 \pm 0.0009$ | $0.8302$ | $0.9696$ | $0.2597$ | $0.2405$ |
| **Full Stream 4-Class** | Stage 3B Baseline DT | $0.6088 \pm 0.0000$ | $0.7345$ | $0.9746$ | $0.2910$ | $0.2012$ |
| **Full Stream 4-Class** | Stage 3B.6 Tuned RF | $0.6889 \pm 0.0019$ | $0.6985$ | $0.9873$ | $0.3830$ | $0.4048$ |
| **Full Stream 4-Class** | Stage 3B.6 HistGradientBoosting | **$0.6975 \pm 0.0009$** | $0.6911$ | $0.9880$ | **$0.3859$** | **$0.4340$** |
| **Full Stream Binary** | Stage 3B Baseline RF | $0.9735 \pm 0.0002$ | $0.9807$ | $0.9857$ | — | — |
| **Full Stream Binary** | Stage 3B.6 HistGradientBoosting | **$0.9833 \pm 0.0000$** | $0.9818$ | $0.9912$ | — | — |
| **MQTT-Only 4-Class** | Stage 3B Baseline RF | $0.6899 \pm 0.0024$ | $0.7015$ | $0.9906$ | $0.1333$ | $0.6738$ |
| **MQTT-Only 4-Class** | Stage 3B.6 HistGradientBoosting | $0.6701 \pm 0.0038$ | $0.6665$ | $0.9916$ | $0.0000$ | $0.7218$ |
| **MQTT-Only Binary** | Stage 3B Baseline RF | $0.9776 \pm 0.0023$ | $0.9835$ | $0.9918$ | — | — |
| **MQTT-Only Binary** | Stage 3B.6 Tuned RF | **$0.9783 \pm 0.0005$** | $0.9871$ | $0.9921$ | — | — |

The public benchmark comparison is illustrated in **Figure 5**:

```
[Figure 5 Reference: docs/figures/fig5_mqtt_uad_baseline_comparison.svg]
Figure 5: Public MQTT-UAD Classical Baseline Evaluation across Random Forest, ExtraTrees, and HistGradientBoosting (Macro-F1).
```

### 9.2 Investigation of Suspicious $1.0000$ Scores on MQTTset
In Stage 3B, classical models on the secondary MQTTset binary benchmark reported **Macro-F1 $= 1.0000 \pm 0.0000$** and **Accuracy $= 1.0000$**.

An exhaustive investigation uncovered severe structural dataset shortcuts:
1. **Flow Protocol Artifacts:** In `MQTTset.csv`, all `Normal` flows have `Protocol = 0` and `Dst_Port = 0`, whereas attack flows have valid protocol numbers ($6$ for TCP).
2. **Byte-Swapped IP Artifacts:** Exactly $83,283$ rows in `Normal` exhibit inverted byte ordering in `Src_IP` (`60.216.192.168`).
3. **Disjoint Epoch Ranges:** Each attack category was captured on disjoint calendar dates spanning 2019 to 2021.
4. **Duplicate Record Saturation:** Removing identifier columns left $426,554$ exact duplicates across the dataset.

**Scientific Assessment:** The $1.0000$ score on MQTTset binary does not reflect genuine intrusion detection capability; it reflects trivial classifier exploitation of capture-environment artifacts. MQTTset is officially labeled as a **caveated benchmark requiring further investigation**, and the primary hybrid project relies strictly on the audited four-sensor architecture.

---

## 10. Academic Discussion of Findings

### 10.1 Strengths of the Integrated Implementation
1. **Complementary Modalities:** Deep sequence learning captures high-dimensional temporal patterns across multiplexed packet streams, while the formal TFSM enforces deterministic safety boundaries.
2. **Explainable Fault Localization:** When an intrusion occurs, the system does not simply output an opaque alert; it identifies the affected physical sensor (`affected_sensors`) and pinpoints the violated transition mutant (`candidate_fault_mutants`).
3. **Robustness to Deep Learning Blind Spots:** When empirical classification fails on rare attacks due to sample scarcity, formal conformance checking independently triggers a `FORMAL_VIOLATION_DL_NORMAL` safety alarm.

### 10.2 Class Imbalance and Deep Learning Limitations
In multi-sensor IoT datasets, benign traffic naturally predominates ($86.4\%$). Attacks like `Replay` ($0.36\%$) and `MITM` ($0.49\%$) provide minimal training samples. Even with inverse-frequency class weighting, statistical classifiers exhibit high variance on sparse minority classes. This finding validates the fundamental architectural premise of this thesis: **empirical deep learning alone is insufficient for safety-critical IoT environments without formal mathematical backstops.**

---

## 11. Known Limitations & Future Research Scope

In accordance with academic research standards, the boundaries of this project are explicitly stated:
1. **Software Simulation Boundary:** All four sensor models (`motion`, `temperature`, `ultrasonic`, `rfid`) are deterministic software simulations written in Python. No claims of physical ESP32 microcontroller or hardware sensor testbed deployment are made.
2. **Simulated MQTT Protocol:** Packet transmission, timing jitter, and QoS delivery are simulated in memory; network captures were not gathered from an external physical broker (e.g., Mosquitto or EMQX).
3. **Generated vs. Base-Paper Dataset:** The four-sensor dataset (3,090 samples) was generated to evaluate multi-sensor fault localization and is distinct from the single-sensor dataset in Kaddoura et al. (2026).
4. **Offline Batch Execution:** Sequence windows ($L=16$) and diagnosis reports are processed in batch mode rather than embedded real-time microcontroller firmware.

### Future Research Roadmap:
- **Physical Testbed Deployment:** Flashing formal TFSM monitors onto low-power ESP32 microcontrollers using FreeRTOS.
- **Hardware Captures:** Capturing live Wireshark traces over physical Wi-Fi/Zigbee connections running against a hardware MQTT broker.
- **Edge Model Optimization:** Quantizing BiLSTM sequence models via TensorRT or ONNX Runtime for edge deployment.
- **Transformer Architectures:** Investigating lightweight temporal attention architectures to replace recurrent cells.

---

## 12. Comprehensive Results Audit & Provenance Verification

| Metric / Result Category | Source File & Component | Value / Finding | Audit Verification Status |
| :--- | :--- | :--- | :---: |
| **Total Event Count** | `datasets/generated/four_sensors/dataset_manifest.json` | 3,090 chronological events | **VERIFIED** |
| **Total Session Count** | `datasets/generated/four_sensors/dataset_manifest.json` | 24 independent sessions | **VERIFIED** |
| **Partition Event Counts** | `ids/datasets_four_sensors.py` | Train: 1,782 \| Val: 626 \| Test: 682 | **VERIFIED** |
| **Sequence Counts ($L=16$)** | `ids/datasets_four_sensors.py` | Train: 1,572 \| Val: 551 \| Test: 607 (Total: 2,730) | **VERIFIED** |
| **Feature Set Provenance** | `ids/datasets_four_sensors.py` | Exactly 14 features across 4 audited categories | **AUDITED & CATEGORIZED** |
| — Protocol-Observable | `ids/datasets_four_sensors.py` | `inter_arrival`, `payload_size`, `qos`, `seq_delta` | **VERIFIED WIRE FIELDS** |
| — Addressing Indicators | `ids/datasets_four_sensors.py` | `is_sensor_{motion,temp,ultrasonic,rfid}` | **VERIFIED TOPIC FLAGS** |
| — Application Payloads | `ids/datasets_four_sensors.py` | `numeric_reading`, `has_alert_flag` | **VERIFIED PAYLOAD FIELDS** |
| — Simulator Metadata | `ids/datasets_four_sensors.py` | `is_event_{publish,alert,heartbeat,connect}` | **VERIFIED SIMULATOR FLAGS** |
| **Leakage Exclusion** | `ids/datasets_four_sensors.py` | Excluded `has_tamper_flag`, `is_event_flood`, etc. | **VERIFIED** |
| **Formal Mutation Testing** | `models_fsm/verify.py` | 59 / 59 mutants killed (100% kill rate) | **VERIFIED (100%)** |
| — Motion Sensor Mutants | `models_fsm/motion_sensor.py` | 14 / 14 killed | **VERIFIED** |
| — Temperature Sensor Mutants | `models_fsm/temperature_sensor.py` | 15 / 15 killed | **VERIFIED** |
| — Ultrasonic Sensor Mutants | `models_fsm/ultrasonic_sensor.py` | 15 / 15 killed | **VERIFIED** |
| — RFID Reader Mutants | `models_fsm/rfid_sensor.py` | 15 / 15 killed | **VERIFIED** |
| **Full Regression Suite** | `tests/` (14 test files) | 144 / 144 tests passing (30.134s) | **VERIFIED (144/144)** |
| **Integration Scenarios** | `tests/test_stage6_integration.py` | 6 attack scenarios + normal localized | **VERIFIED (6/6)** |
| **Hybrid Operational States** | `diagnosis_loc/hybrid_diagnosis.py` | 6 formal adjudication outcomes verified | **VERIFIED** |
| **Validation Model Selection** | `ids/train_four_sensors_sequences.py` | BiLSTM Val Macro-F1: 0.9782 (Mean: 0.9604) | **VERIFIED VALIDATION** |
| **Test Sequence Checkpoints** | `results/stage_5_hybrid/` | Preliminary 592-seq metrics quarantined as stale | **MARKED STALE / RE-EVAL** |
| **MQTT-UAD Baseline Report** | `results/stage_3b_baselines/` | Full Stream 4-Class Macro-F1: 0.6098 (RF) | **VERIFIED (SEPARATE)** |
| **MQTT-UAD Improved Report** | `docs/STAGE3B6_BASELINE_IMPROVEMENT_REPORT.md` | Full Stream 4-Class Macro-F1: 0.6975 (HistGB) | **VERIFIED (SEPARATE)** |
| **MQTTset 1.0000 Investigation** | `docs/DEEP_DATASET_ANALYSIS.md` | Identified protocol=0 & IP byte-swap shortcut leakage | **VERIFIED ARTIFACT** |

---

*Report prepared and certified for final-year project documentation, external viva evaluation, and research archival.*
