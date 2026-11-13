# Attack Diagnosis and Localization in IoT Intrusion Detection Using Formal Modeling and Deep Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-144%2F144%20Passing-brightgreen.svg)]()
[![Mutants Killed](https://img.shields.io/badge/Formal%20Mutants-59%2F59%20Killed%20(100%25)-brightgreen.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-BiLSTM%20%2B%20Timed%20FSM-orange.svg)]()

> **Final-Year Project:** *Attack Diagnosis and Localization in IoT Intrusion Detection Using Formal Modeling and Deep Learning*  
> **Repository:** `iot-ids-diagnosis`  
> **Branch:** `simulation-setup`  
> **Core Result:** Complete two-tier intrusion detection and root-cause fault localization framework integrating deep sequence modeling with deterministic Timed Finite State Machines across four software-simulated IoT sensors.

---

## 1. Project Objective & Problem Statement

Conventional deep learning intrusion detection systems (IDS) in IoT classify network traffic statistically but function as opaque black boxes: they fail to explain the operational root cause, cannot pinpoint the specific sensor or protocol transition affected, and are vulnerable to empirical classification errors on minority attack classes.

This project designs and implements an integrated, explainable **two-tier IoT Intrusion Detection and Fault Localization framework**:
1. **Tier 1 (Statistical Sequence Modeling):** Employs Bidirectional Long Short-Term Memory (`BiLSTM`) networks over sliding packet windows ($L=16$) to detect temporal and multi-packet attack signatures.
2. **Tier 2 (Deterministic Conformance Verification):** Enforces mathematically rigorous Timed Mealy Finite State Machines (`TFSM`) per sensor to detect protocol deviations, clock timing violations, sequence tampering, and physical domain bounds violations.
3. **Hybrid Adjudication & Fault Localization:** Resolves statistical probabilities and deterministic formal proofs into transparent verdicts (`CONSENSUS`, `FORMAL_VIOLATION_DL_NORMAL`, `STATISTICAL_ANOMALY`, `NOMINAL`) while mapping protocol violations to candidate mutant fault hypotheses.

This work directly extends the foundational formal-modeling-assisted IDS research of:
> Y. Kaddoura, K. El-Fakih, I. Zualkernan, A. Bhanpurawala,  
> *Deep learning-based intrusion detection for IoT – an approach assisted by formal modeling and mutation testing*,  
> **Internet of Things**, Vol. 38 (2026), 101958. https://doi.org/10.1016/j.iot.2026.101958  
> *(See also Bhanpurawala et al., arXiv:2210.05623, for attacks-as-mutants modeling)*.

### Key Contributions Beyond the Base Paper
- **Multi-Sensor Ecosystem:** Extends the single motion sensor model to four heterogeneous IoT sensors (`motion`, `temperature`, `ultrasonic`, `rfid`).
- **59 Mutants Verified:** Implements and mathematically proves 59 distinct fault mutants (100% killed).
- **Zero-Leakage 14-Feature Sequence Pipeline:** Formulates a rigorous observable feature extractor strictly quarantining ground-truth labels and simulator metadata.
- **Root-Cause Fault Localization:** Maps formal anomalies directly to candidate transition and state mutants (`candidate_faults`).
- **Resilient Hybrid Adjudication:** Guarantees that deep learning failures never suppress formal safety proofs (`MODEL_UNAVAILABLE` fallback preserves 100% formal detection).

---

## 2. System Architecture

```
                             Four Software-Simulated Sensors
                         (Motion, Temperature, Ultrasonic, RFID)
                                           |
                                           v
                           Unified Discrete-Event Simulator
                          (Monotonic Clock, Session Jitter)
                                           |
                                           v
                           Sensor-Aware Attack Injector
                 (DoS, Replay, MITM, DataFalsification, Delay, Spoof)
                                           |
                                           v
                              Observable MQTT Packet Stream
                                           |
                 +-------------------------+-------------------------+
                 |                                                   |
                 v                                                   v
       [ Feature Extractor ]                               [ Sensor Monitors ]
       14 Observable Features                              Per-Sensor State & Clock
       Zero-Leakage Guarantee                              Conformance Checking
                 |                                                   |
                 v                                                   v
       [ Sequence Builder ]                                [ Formal TFSM Engine ]
       Sliding Windows (L=16)                              Timing & Bounds Invariants
       Session-Isolated Splits                             Candidate Mutant Mapping
                 |                                                   |
                 v                                                   v
       [ Tier 1: BiLSTM ]                                  [ Tier 2: Formal Proof ]
       Statistical Attack Category                         PASS / FAIL Conformance
       Softmax Confidence Score                            Physical Deviation Types
                 |                                                   |
                 +-------------------------+-------------------------+
                                           |
                                           v
                           [ Hybrid Adjudication Arbiter ]
                           - Reconciles empirical & formal proofs
                           - Preserves independent evidence fields
                                           |
                                           v
                             Unified Hybrid Diagnosis Report
```

---

## 3. Four Simulated Sensor Models

Each sensor is formalized as a deterministic Timed Finite State Machine $\mathcal{M} = (S, s_0, I, O, C, T)$ with explicit clock bounds, state invariants, and output alphabets:

| Sensor | Physical Domain | States & Behavior | Keepalive Timeout | Mutants Defined | Mutants Killed |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Motion Sensor** (`PIR`) | Binary occupancy (`0.0` / `1.0`) | Idle $\leftrightarrow$ Motion Detected $\rightarrow$ Alert | $1.5\text{ s}$ | 14 | **14 / 14 (100%)** |
| **Temperature Sensor** | Ambient telemetry ($15.0^\circ\text{C} - 35.0^\circ\text{C}$) | Periodic Sense $\rightarrow$ Delta Publish | $2.0\text{ s}$ | 15 | **15 / 15 (100%)** |
| **Ultrasonic Sensor** | Obstacle distance ($2.0\text{ cm} - 400.0\text{ cm}$) | Ranging $\rightarrow$ Proximity Alert | $2.0\text{ s}$ | 15 | **15 / 15 (100%)** |
| **RFID Access Reader** | Contactless card UID credentials | Card Tap $\rightarrow$ Authenticate $\rightarrow$ Grant/Deny | $3.0\text{ s}$ | 15 | **15 / 15 (100%)** |
| **TOTAL** | | | | **59** | **59 / 59 (100%)** |

All 59 mutants are verified via formal mutation test suites in [`models_fsm/verify.py`](models_fsm/verify.py).

---

## 4. Unified Simulator & Six Attack Scenarios

The unified simulator ([`simulation/unified_simulator.py`](simulation/unified_simulator.py)) maintains a discrete-event monotonic clock running multi-session network traces with variable inter-arrival jitter, QoS levels, and message payload structures.

The attack injector ([`simulation/attack_injector.py`](simulation/attack_injector.py)) introduces six sensor-aware attack scenarios:

1. **Denial-of-Service (`DoS`):** High-frequency packet flood surges intended to exhaust sensor queues and broker buffers.
2. **Replay Attack (`Replay`):** Captures historical sensor telemetry and re-injects stale sequence numbers with transmission delays.
3. **Man-in-the-Middle (`MITM`):** Intercepts and tampers with payload data in transit, creating semantic contradictions between telemetry and reported status.
4. **Data Falsification (`DataFalsification`):** Corrupts raw readings to violate physical sensor domain bounds (e.g., negative ultrasonic distance or impossible temperatures).
5. **Message Delay (`Delay`):** Artificially stalls legitimate telemetry beyond protocol keepalive timeout thresholds.
6. **Identity Spoofing (`IdentitySpoofing`):** Transmits forged RFID card UID credentials attempting unauthorized access state transitions.

---

## 5. Dataset Generation & 14 Observable Features

The 24-session unified dataset ([`datasets/generated/four_sensors/four_sensors_dataset.csv`](datasets/generated/four_sensors/four_sensors_dataset.csv)) contains **3,090 chronological packet rows** partitioned strictly by session to prevent cross-session leakage:

- **Train Split (Sessions 0–13):** 1,782 rows $\rightarrow$ 1,572 sequence windows ($L=16$).
- **Validation Split (Sessions 14–18):** 626 rows $\rightarrow$ 551 sequence windows ($L=16$).
- **Test Split (Sessions 19–23):** 682 rows $\rightarrow$ 607 sequence windows ($L=16$).

### Strict Zero-Leakage Observable Feature Set
All artificial injector watermarks (`has_tamper_flag`, `is_event_flood`) and internal simulator state variables (`observed_state`, `output_symbol`) are strictly quarantined. Exactly **14 observable features** available from raw network traffic are used:

| Feature Name | Description | Provenance / Type |
| :--- | :--- | :--- |
| `inter_arrival` | Time delta between consecutive packets ($\Delta t$) | Network timing |
| `payload_size` | Length of MQTT payload in bytes | Packet header |
| `qos` | MQTT Quality of Service flag ($0, 1, 2$) | MQTT header |
| `seq_delta` | Monotonic sequence number difference ($\Delta\text{seq}$) | Network protocol |
| `is_sensor_motion` | One-hot indicator: Motion Sensor topic | Network routing |
| `is_sensor_temp` | One-hot indicator: Temperature Sensor topic | Network routing |
| `is_sensor_ultrasonic` | One-hot indicator: Ultrasonic Sensor topic | Network routing |
| `is_sensor_rfid` | One-hot indicator: RFID Sensor topic | Network routing |
| `is_event_publish` | One-hot indicator: MQTT `PUBLISH` frame | MQTT frame type |
| `is_event_alert` | One-hot indicator: MQTT alert topic suffix | MQTT topic syntax |
| `is_event_heartbeat` | One-hot indicator: MQTT ping/heartbeat frame | MQTT frame type |
| `is_event_connect` | One-hot indicator: MQTT `CONNECT`/handshake frame | MQTT frame type |
| `numeric_reading` | Parsed physical floating-point payload value | Packet body |
| `has_alert_flag` | Binary indicator parsed from payload message text | Packet body |

Normalizers (`MaskedStandardScaler`) are fitted **strictly on the training split** and applied to validation and test splits without lookahead.

---

## 6. Deep Sequence Models: LSTM & BiLSTM

Deep learning sequence architectures in [`ids/models_sequence.py`](ids/models_sequence.py) process sequence windows of shape `(batch, 16, 14)`:

- **Optimizer:** `AdamW` (learning rate: $10^{-3}$, weight decay: $10^{-4}$).
- **Early Stopping:** Monitored on **Validation Macro-F1** with patience of 5 epochs (maximum 50 epochs).
- **Class Imbalance Handling:** Inverse-frequency class weighting applied during cross-entropy loss computation.
- **Production Architecture:** `BiLSTMClassifier` achieved the highest validation Macro-F1 (**$0.9782$**), outperforming unidirectional LSTM ($0.9412 \pm 0.0210$).

---

## 7. Formal Diagnosis & Hybrid Adjudication

The hybrid diagnosis engine ([`diagnosis_loc/hybrid_diagnosis.py`](diagnosis_loc/hybrid_diagnosis.py)) executes both tiers and adjudicates their findings into six formal outcomes:

| Operational Outcome | Formal TFSM Tier | Deep Learning Tier | Diagnostic Action & System Verdict |
| :--- | :---: | :---: | :--- |
| `CONSENSUS` | Violation Detected | Attack Detected | Mutual concurrence. Full alert with DL classification and formal mutant localization. |
| `FORMAL_VIOLATION_DL_NORMAL` | Violation Detected | Normal | **Safety-critical override.** Formal invariant proof takes precedence over statistical DL. |
| `STATISTICAL_ANOMALY` | Conforming | Attack Detected | Early warning advisory. Flagged for review; formal protocol bounds still hold. |
| `NOMINAL` | Conforming | Normal | Legitimate benign network operation confirmed by both tiers. |
| `INSUFFICIENT_SEQUENCE` | Conforming / Evaluated | Insufficient Data ($N < 16$) | Sequence window too short; formal diagnosis runs autonomously. |
| `MODEL_UNAVAILABLE` | Conforming / Evaluated | Checkpoint Disabled | Neural weights absent; formal TFSM operates with **zero degradation**. |

---

## 8. Testing & Verification Summary

The complete codebase is validated by **144 automated unit tests across 14 test suites** and formal mutation testing:

```
======================================================================
1.  tests/test_fsm_sensors.py           21/21 PASS  [Four-sensor TFSM specifications]
2.  tests/test_unified_simulator.py       8/8 PASS  [Discrete-event clock & MQTT stream]
3.  tests/test_attack_injector.py         8/8 PASS  [Six attack injection scenarios]
4.  tests/test_phase3_data.py           10/10 PASS  [MQTT dataset schema validation]
5.  tests/test_stage3_dataset.py          9/9 PASS  [Four-sensor dataset generation]
6.  tests/test_stage3b_baselines.py       3/3 PASS  [Tabular machine learning baselines]
7.  tests/test_stage3b6_improvements.py   3/3 PASS  [Feature engineering & data metrics]
8.  tests/test_stage3b6_runtime.py        5/5 PASS  [Execution latency & memory limits]
9.  tests/test_stage3c_sequences.py       8/8 PASS  [Sequence windowing & batch loaders]
10. tests/test_unified_diagnosis.py     16/16 PASS  [Formal conformance & fault localization]
11. tests/test_stage5_dataset.py        14/14 PASS  [14-feature zero-leakage pipeline]
12. tests/test_stage5_training.py         8/8 PASS  [Multi-seed sequence model training]
13. tests/test_stage5_hybrid.py         15/15 PASS  [Two-tier adjudication & fallback]
14. tests/test_stage6_integration.py    16/16 PASS  [End-to-end integration & scenario audit]
----------------------------------------------------------------------
TOTAL AUTOMATED TESTS:                 144/144 PASS (100% Pass Rate)
FORMAL MUTANTS KILLED:                   59/59 KILLED (100% Kill Rate)
======================================================================
```

---

## 9. Quickstart & Execution Guide

### Environment Setup
Use Python 3.10 or newer in Windows Command Prompt (`cmd.exe`):

```cmd
:: 1. Navigate to project repository root
cd /d c:\Users\cheru\Downloads\iot-ids-diagnosis\iot-ids-diagnosis

:: 2. Activate virtual environment
call .venv_stage3c\Scripts\activate

:: 3. Install dependencies
pip install -r requirements.txt
```

### Verification & Demonstration Commands

```cmd
:: Run the full 144-test regression suite
python -m unittest discover -s tests -v

:: Run formal TFSM mutation testing (verifies 59/59 mutants killed)
python -m models_fsm.verify

:: Run Stage 6 end-to-end integration audit
python -m unittest tests.test_stage6_integration -v

:: Run the top-level full system demonstration harness
python run_full_system.py
```

---

## 10. Research Scope & Limitations

In the interest of rigorous scientific integrity, the following boundaries govern this project:
1. **Software Simulation Scope:** All four sensor models (`motion`, `temperature`, `ultrasonic`, `rfid`) are deterministic Python software simulations conforming to Timed FSM definitions. No physical ESP32 microcontrollers or hardware sensor probes are used.
2. **Simulated MQTT Protocol:** MQTT packet delivery, timestamps, QoS, and topics are simulated within the discrete-event simulator; traffic is not captured from an external physical broker (e.g., Mosquitto or EMQX).
3. **Generated vs Base Paper Dataset:** The four-sensor dataset (3,090 samples) was synthesized specifically to test multi-sensor fault localization across heterogeneous devices; it is not the single-sensor dataset from the base paper.
4. **Softmax Output Semantics:** Softmax output scores from Tier 1 neural networks reflect empirical model confidence, **not calibrated Bayesian probabilities**. The hybrid arbiter uses them strictly as empirical evidence and never elevates them to formal mathematical proofs.
5. **No Production / Real-Time Claims:** The implementation is a research testbed designed for algorithmic validation and formal verification, not a certified production cyber-physical IDS.

---

## 11. Project Structure

```text
iot-ids-diagnosis/
│
├── models_fsm/                 # Formal Timed FSM Models & Mutation Testing
│   ├── tfsm.py                 # Core Timed FSM Engine & Clock Evaluator
│   ├── motion_sensor.py        # Motion Sensor Model (14 Mutants)
│   ├── temperature_sensor.py   # Temperature Sensor Model (15 Mutants)
│   ├── ultrasonic_sensor.py    # Ultrasonic Sensor Model (15 Mutants)
│   ├── rfid_sensor.py          # RFID Reader Model (15 Mutants)
│   └── verify.py               # Mutation Verification Test Harness (59/59)
│
├── simulation/                 # Discrete-Event Network Simulation
│   ├── unified_simulator.py    # Multi-Sensor Simulator with Monotonic Clock
│   └── attack_injector.py      # Sensor-Aware Injector for 6 Attack Scenarios
│
├── features/                   # Dataset Building & Feature Engineering
│   └── build_unified_dataset.py # 24-Session Dataset Generator (3,090 Rows)
│
├── ids/                        # Deep Learning Sequence Models & Training
│   ├── models_sequence.py      # LSTMClassifier & BiLSTMClassifier Architectures
│   ├── datasets_four_sensors.py # 14-Feature Zero-Leakage Dataset & Sequence Loader
│   ├── datasets_real.py        # MaskedStandardScaler & Feature Engineering
│   └── train_four_sensors_sequences.py # Multi-Seed Training Engine & Early Stopping
│
├── diagnosis_loc/              # Formal Conformance & Hybrid Diagnosis
│   ├── unified_diagnosis.py    # Per-Sensor Monitor & Mutant Fault Localizer
│   └── hybrid_diagnosis.py     # Two-Tier Hybrid Adjudication Arbiter
│
├── datasets/
│   └── generated/four_sensors/ # Generated Multi-Sensor Dataset & Manifest
│       ├── four_sensors_dataset.csv
│       └── dataset_manifest.json
│
├── docs/                       # Comprehensive Architecture & Audit Reports
│   ├── FOUR_SENSOR_INTEGRATED_SYSTEM.md  # Complete System Specification
│   ├── STAGE6_FINAL_INTEGRATION_REPORT.md # Final Integration & Audit Report
│   ├── STAGE5_1_DATASET_REPORT.md         # Feature Provenance & Leakage Audit
│   └── ...
│
├── tests/                      # Automated Unit & Integration Regression Suite
│   ├── test_fsm_sensors.py     # 21 Tests: TFSM models & transitions
│   ├── test_unified_simulator.py # 8 Tests: discrete-event simulator
│   ├── test_attack_injector.py # 8 Tests: attack injection mechanics
│   ├── test_phase3_data.py     # 10 Tests: raw dataset schema validation
│   ├── test_stage3_dataset.py  # 9 Tests: four-sensor dataset builder
│   ├── test_stage3b_baselines.py # 3 Tests: tabular ML baselines
│   ├── test_stage3b6_improvements.py # 3 Tests: feature engineering
│   ├── test_stage3b6_runtime.py # 5 Tests: runtime latency & limits
│   ├── test_stage3c_sequences.py # 8 Tests: sequence windows & batching
│   ├── test_unified_diagnosis.py # 16 Tests: formal diagnosis engine
│   ├── test_stage5_dataset.py  # 14 Tests: 14 observable features loader
│   ├── test_stage5_training.py # 8 Tests: multi-seed training & early stopping
│   ├── test_stage5_hybrid.py   # 15 Tests: two-tier hybrid adjudication
│   └── test_stage6_integration.py # 16 Tests: end-to-end integration audit
│
├── run_full_system.py          # Top-Level Integrated Demonstration CLI
├── requirements.txt            # Python Dependencies
└── .gitignore                  # Git Exclusion Rules (Protects Venvs & Secrets)
```

---

## 12. Authors & Acknowledgments

- **Author:** Cherukuri Ram Charan ([@CherukuriRamCharan](https://github.com/CherukuriRamCharan))
- **Project Title:** *Attack Diagnosis and Localization in IoT Intrusion Detection Using Formal Modeling and Deep Learning*
- **Academic Foundation:** Extends Kaddoura et al. (*Internet of Things*, Elsevier, 2026).
