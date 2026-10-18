# Attack Diagnosis and Localization in IoT Intrusion Detection
## Using Formal Modeling and Deep Learning

**B.Tech CSE Final-Year Project** — Ch. Ram Charan, K. Varun, A. Varun Satya

> Reference paper: Kaddoura et al. (2026) — *"Deep learning-based intrusion detection for IoT — an approach assisted by formal modeling and mutation testing"*, Internet of Things.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Environment Setup](#4-environment-setup)
5. [Running the Pipeline](#5-running-the-pipeline)
6. [Dataset](#6-dataset)
7. [Formal Models (EFSM)](#7-formal-models-efsm)
8. [Attack Scenarios](#8-attack-scenarios)
9. [Feature Extraction](#9-feature-extraction)
10. [Machine Learning Models](#10-machine-learning-models)
11. [Diagnosis and Localization](#11-diagnosis-and-localization)
12. [Evaluation Results](#12-evaluation-results)
13. [Testing](#13-testing)
14. [Limitations](#14-limitations)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Project Overview

This project extends a formal-modeling-assisted deep-learning approach
for IoT intrusion detection to include **attack diagnosis and localization**.

The system:

- Models IoT device behaviour using Extended Finite-State Machines (EFSMs)
- Generates four attack simulations (replay, data falsification, message manipulation, battery draining)
- Captures MQTT traffic for both normal and attack scenarios
- Extracts network features (packet length, ports, TCP flags, IAT, payload size)
- Trains three classifiers: Random Forest baseline, LSTM, and Transformer
- Applies rule-based diagnosis using payload and packet-rate evidence
- Produces per-capture diagnosis records with localization evidence

**Attack classes** (from external dataset labels):

| Class | Description |
|---|---|
| Benign | Normal IoT network traffic |
| DoS | TCP SYN flooding |
| Mirai | Botnet (ACK flood, HTTP flood, UDP flood, host brute-force) |
| MITM | ARP spoofing |
| Scan | Port/host scanning |

**Simulated scenarios** (from local captures):

| Scenario | Simulation Type |
|---|---|
| Replay-like | Payload-level repeated publishing |
| Data falsification | Out-of-range temperature values |
| Message manipulation | Application-level forwarding with modification |
| Excessive communication | High-frequency publish burst |

---

## 2. Architecture

```text
IoT MQTT Simulation
  (temperature_device.py, motion_device.py)
        │
        ▼
  MQTT Broker (Mosquitto :1883)
        │
        ├── Attack Injection
        │   (replay, falsification, mitm, battery)
        │
        ▼
  PCAP Captures           External Dataset (42 PCAPs)
  captures/normal/        data/base_paper/Packets/
  captures/attack/
        │                         │
        ▼                         ▼
  MQTT Feature Extraction   Packet Feature Extraction
  (TShark)                  (add_iat_features.py)
        │                         │
        ├── mqtt_traffic_features.csv
        ├── mqtt_payload_features.csv
        └── all_packet_features_iat.csv
                    │
        ┌───────────┼────────────┐
        ▼           ▼            ▼
  RF Baseline    LSTM        Transformer
  (train_rf_    (train_      (train_
   5class.py)    lstm.py)     transformer.py)
        │           │            │
        └───────────┼────────────┘
                    ▼
          Combined Diagnosis Report
          (EFSM violations + ML predictions
           + payload rules + rate rules)
                    │
                    ▼
          Localization Record
          (observed IP/port endpoint)
```

**EFSM models** run independently as rule-based detectors (in `models_fsm/`).
They provide complementary evidence to the ML classifiers.

---

## 3. Repository Structure

```
IoT-Attack-Diagnosis-Localization/
├── captures/
│   ├── normal/          MQTT captures of normal device operation
│   └── attack/          MQTT captures of simulated attacks
├── data/
│   └── base_paper/
│       ├── Packets/     42 PCAP files (external dataset)
│       ├── all_packet_features.csv       (407 MB — not in Git)
│       ├── all_packet_features_iat.csv   (generated — not in Git)
│       ├── add_iat_features.py           IAT enrichment script
│       └── extract_dataset_features.py  TShark extraction script
├── docs/
│   ├── attack_scenarios/attack_scenario_log.csv
│   ├── traffic_mapping.csv
│   └── traceability_matrix.md           Paper-to-project traceability
├── features/
│   ├── extract_mqtt_traffic.py
│   ├── extract_mqtt_payloads.py
│   ├── mqtt_traffic_features.csv
│   └── mqtt_payload_features.csv
├── ids/
│   ├── train_base_paper_model.py   RF binary baseline
│   ├── train_rf_5class.py          RF 5-class with IAT
│   ├── train_lstm.py               LSTM pipeline
│   ├── train_transformer.py        Transformer pipeline
│   ├── predict_capture_attacks.py  Predict on all 42 captures
│   ├── run_combined_diagnosis.py   ML + rule diagnosis per capture
│   ├── create_combined_diagnosis.py  MQTT diagnosis report
│   ├── evaluate_rules_on_captures.py  Payload rule evaluation
│   ├── evaluate_packet_rates.py   Rate rule evaluation
│   ├── final_evaluation.py         Capture-level evaluation
│   └── diagnosis_rules.py          Rule helpers
├── models/
│   ├── base_paper_rf.joblib        Binary RF (trained)
│   ├── rf_5class.joblib            5-class RF (run train script)
│   ├── lstm_ids.pt                 LSTM checkpoint (run train script)
│   └── transformer_ids.pt         Transformer checkpoint (run train script)
├── models_fsm/
│   ├── temperature_efsm.py
│   ├── temperature_model.md
│   ├── temperature_state_mapping.csv
│   ├── motion_efsm.py              NEW
│   └── motion_model.md             NEW
├── results/                        All evaluation JSON/CSV files
├── simulation/
│   ├── temperature_device.py
│   ├── motion_device.py
│   ├── mqtt_config.py
│   └── attack_scenarios/
│       ├── replay_attack.py
│       ├── data_falsification.py
│       ├── mitm_simulation.py
│       └── battery_draining_simulation.py
├── tests/
│   ├── test_diagnosis_rules.py     Existing tests
│   └── test_extended.py            Extended tests (EFSM, schema, model)
├── tools/
│   └── run_project_diagnostics.py  End-to-end offline runner
├── diagnosis_config.py             Shared rule settings
└── README.md
```

---

## 4. Environment Setup

**Requirements:** Python 3.11, Wireshark/TShark on PATH, Mosquitto (optional).

```powershell
# From the project root
python -m venv .venv
.venv\Scripts\pip install pandas numpy scikit-learn torch joblib paho-mqtt pytest
```

Verify TShark is available:
```powershell
tshark --version
```

---

## 5. Running the Pipeline

All commands are run from the project root in PowerShell.

### Step 0 — Verify existing state (offline, no Mosquitto needed)
```powershell
.venv\Scripts\python.exe tools\run_project_diagnostics.py
```

### Step 1 — Run existing tests
```powershell
.venv\Scripts\python.exe -m pytest tests\ -v
```

### Step 2 — Generate IAT-enriched dataset (~20-30 min, 407 MB input)
```powershell
.venv\Scripts\python.exe data\base_paper\add_iat_features.py
```
Output: `data/base_paper/all_packet_features_iat.csv`

### Step 3 — Train 5-class Random Forest baseline
```powershell
.venv\Scripts\python.exe ids\train_rf_5class.py
```
Output: `models/rf_5class.joblib`, `results/rf_5class_evaluation.json`

### Step 4 — Train LSTM
```powershell
.venv\Scripts\python.exe ids\train_lstm.py
```
Output: `models/lstm_ids.pt`, `results/lstm_evaluation.json`

### Step 5 — Train Transformer
```powershell
.venv\Scripts\python.exe ids\train_transformer.py
```
Output: `models/transformer_ids.pt`, `results/transformer_evaluation.json`

### Step 6 — Run binary RF predictions on all captures
```powershell
.venv\Scripts\python.exe ids\predict_capture_attacks.py
```

### Step 7 — Run combined ML + rule diagnosis (42 captures)
```powershell
.venv\Scripts\python.exe ids\run_combined_diagnosis.py
```

### Step 8 — Capture-level evaluation
```powershell
.venv\Scripts\python.exe ids\final_evaluation.py
```

### Step 9 — MQTT rule-based diagnosis (6 local captures)
```powershell
.venv\Scripts\python.exe features\extract_mqtt_traffic.py
.venv\Scripts\python.exe features\extract_mqtt_payloads.py
.venv\Scripts\python.exe ids\evaluate_rules_on_captures.py
.venv\Scripts\python.exe ids\evaluate_packet_rates.py
.venv\Scripts\python.exe ids\create_combined_diagnosis.py
```

### Predict only (skip retraining)
```powershell
.venv\Scripts\python.exe ids\train_lstm.py --predict
.venv\Scripts\python.exe ids\train_transformer.py --predict
```

### Live MQTT test (requires Mosquitto running)
```powershell
.venv\Scripts\python.exe tools\run_project_diagnostics.py --live-mqtt-test
```

---

## 6. Dataset

### External Dataset (base paper)
- **Source:** IoT Network Intrusion Dataset (Kaggle / UNSW)
- **Files:** 42 PCAP captures in `data/base_paper/Packets/`
- **Total packets:** ~2,985,994
- **Classes:** benign (1), dos (6), mirai (14), mitm (6), scan (12) captures
- **Feature CSV:** `data/base_paper/all_packet_features.csv` (407 MB, not in Git)
- **Labels:** derived from capture filenames

### Local MQTT Captures
- 2 normal captures (temperature + motion)
- 4 attack captures (replay, falsification, manipulation, battery)
- Stored in `captures/` (not in Git due to `.gitignore`)

---

## 7. Formal Models (EFSM)

Two EFSM models are implemented:

| Device | File | States | Detectors |
|---|---|---|---|
| Temperature sensor | `models_fsm/temperature_efsm.py` | 7 | replay, falsification, manipulation, excessive-rate |
| Motion sensor | `models_fsm/motion_efsm.py` | 7 | repeated-value (evidence), invalid-value, manipulation, excessive-rate |

Run EFSM self-tests:
```powershell
.venv\Scripts\python.exe models_fsm\temperature_efsm.py
.venv\Scripts\python.exe models_fsm\motion_efsm.py
```

---

## 8. Attack Scenarios

| Scenario ID | Label | Simulation Type | Capture File |
|---|---|---|---|
| ATK-REPLAY-001 | Replay | Payload-level repeat | `replay_temperature_001.pcapng` |
| ATK-FALSIFY-001 | Data Falsification | Altered temperature | `data_falsification_temperature_001.pcapng` |
| ATK-MANIP-001 | Message Manipulation | App-level forwarding | `mitm_simulation.pcapng` |
| ATK-BATTERY-001 | Excessive Communication | High-frequency burst | `battery_draining_temperature_001.pcapng` |

**Important:** These are controlled application-level simulations.
- The MITM simulation is **not** a verified network-path man-in-the-middle attack.
- The battery-draining simulation measures MQTT publish frequency, **not** physical energy.
- Captures are excluded from Git (`.gitignore` rule).

---

## 9. Feature Extraction

### External dataset features
| Feature | Source | Description |
|---|---|---|
| `packet_length` | TShark `frame.len` | Total frame size in bytes |
| `tcp_src_port` | TShark `tcp.srcport` | TCP source port |
| `tcp_dst_port` | TShark `tcp.dstport` | TCP destination port |
| `udp_src_port` | TShark `udp.srcport` | UDP source port |
| `udp_dst_port` | TShark `udp.dstport` | UDP destination port |
| `tcp_syn` | TShark flag | TCP SYN flag |
| `tcp_ack` | TShark flag | TCP ACK flag |
| `tcp_reset` | TShark flag | TCP RST flag |
| `udp_length` | TShark `udp.length` | UDP datagram length |
| `iat_seconds` | computed | Inter-arrival time from previous packet in capture |
| `rolling_10s_pkt_rate` | computed | Packets in trailing 10-second window |

---

## 10. Machine Learning Models

| Model | Script | Classes | Window | Status |
|---|---|---|---|---|
| RF binary | `ids/train_base_paper_model.py` | benign/attack | packet | ✅ Trained |
| RF 5-class | `ids/train_rf_5class.py` | 5 classes | packet | Run script |
| LSTM | `ids/train_lstm.py` | 5 classes | 20 packets | Run script |
| Transformer | `ids/train_transformer.py` | 5 classes | 20 packets | Run script |

All models use **capture-grouped** train/test split to prevent data leakage.
Scaler is fitted on training data only.

---

## 11. Diagnosis and Localization

**Localization levels supported:**

1. ✅ Attack detection (binary: benign vs attack)
2. ✅ Attack-type classification (5 classes)
3. ✅ Network endpoint identification (observed IP/port)
4. ❌ Physical IoT device localization (not supported — requires additional infrastructure)
5. ❌ Attacker attribution (not supported — IP ≠ identity)

**Diagnosis record fields:**
- `capture_file` — source capture
- `dataset_label` / `simulated_scenario` — provenance
- `ml_attack_percent` — fraction of packets predicted attack
- `rule_evidence` — triggered rule signals
- `detection_signals` — named signals (range violation, rate, manipulation)
- `supporting_evidence` — human-readable evidence string
- `localization_status` — level of localization achieved
- `uncertainty_and_limitations` — honest limitations statement

---

## 12. Evaluation Results

**Binary RF (existing, reported honestly):**

| Metric | Benign | Attack | Notes |
|---|---|---|---|
| Precision | 0.41 | 1.00 | Benign precision low due to class imbalance |
| Recall | 0.98 | 0.96 | — |
| F1 | 0.58 | 0.98 | — |
| Accuracy | 96.1% | — | Benign test = same-capture temporal proxy |

**Capture-level:** 100% on 42 captures — **not independent validation** (see `results/final_capture_evaluation.json`).

**5-class RF, LSTM, Transformer:** see `results/` after running respective training scripts.

---

## 13. Testing

```powershell
# All tests
.venv\Scripts\python.exe -m pytest tests\ -v

# Specific test files
.venv\Scripts\python.exe -m pytest tests\test_diagnosis_rules.py -v
.venv\Scripts\python.exe -m pytest tests\test_extended.py -v
```

Test coverage:
- `test_diagnosis_rules.py` — 7 tests: rule helpers, EFSM detectors, signal logic
- `test_extended.py` — 22 tests: both EFSMs, dataset schema, model loading, sequences

---

## 14. Limitations

- **One benign capture** — benign test set is a same-capture temporal proxy; not independent.
- **Labels from filenames** — all packets per capture share a single label; individual packets are not independently labelled.
- **Simulation ≠ real attack** — MITM is application-level; battery draining measures MQTT frequency only.
- **Rate threshold provisional** — 20 publishes / 10 s is calibrated on a single normal capture.
- **Endpoint localization only** — an IP address does not identify a physical device or attacker.
- **42 captures** — insufficient for strong generalisation claims; use results as proof-of-concept.
- **IAT must be generated** — run `add_iat_features.py` before LSTM/Transformer.
- **No GPU required** — all models run on CPU (GPU used automatically if available).

---

## 15. Troubleshooting

| Problem | Solution |
|---|---|
| `tshark: command not found` | Install Wireshark and add to PATH |
| `FileNotFoundError: all_packet_features.csv` | Run `data\base_paper\extract_all_packets.py` first |
| `FileNotFoundError: all_packet_features_iat.csv` | Run `data\base_paper\add_iat_features.py` |
| `FileNotFoundError: models\lstm_ids.pt` | Run `ids\train_lstm.py` first |
| `Connection refused` (MQTT) | Start Mosquitto: `net start mosquitto` |
| Test `test_csv_label_classes` fails | The 407 MB CSV may not be present; it is excluded from Git |
| Out of memory during IAT generation | Reduce `CHUNK` in `add_iat_features.py` from 200_000 to 100_000 |
| LSTM training very slow on CPU | Reduce `MAX_PER_CAPTURE` or `EPOCHS` in `train_lstm.py` |
