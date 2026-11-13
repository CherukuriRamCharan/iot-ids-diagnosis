# Stage 3 Report: Unified Four-Sensor IoT Dataset Generation & Validation

> **Project:** *Attack Diagnosis and Localization in IoT Intrusion Detection Using Formal Modeling and Deep Learning*  
> **Component:** Stage 3 — Unified Dataset Generation & Anti-Leakage Feature Pipeline  
> **Data Nature Disclaimer:** The dataset described and generated in this stage consists of **software-simulated IoT network events** produced by the discrete-event `UnifiedSimulator` and `AttackInjector`. It does **not** represent physical ESP32/hardware captures and is **not** the original base paper's experimental dataset.

---

## 1. Executive Summary

Stage 3 implements a leak-free dataset generation pipeline that translates concurrent multi-sensor simulations into structured, machine-learning-ready datasets:
- **Pipeline Implementation**: [features/build_unified_dataset.py](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/features/build_unified_dataset.py).
- **Dedicated Output Directory**: [datasets/generated/four_sensors/](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/datasets/generated/four_sensors/).
- **Output Files**:
  - `four_sensors_dataset.csv`: Complete dataset with 29 columns (16 observable features, 4 ground-truth labels, 9 metadata columns).
  - `dataset_manifest.json`: Full provenance metadata, schema specification, class and sensor counts per split.
- **Scenario & Session Architecture**:
  - **24 independent simulation sessions** (30.0 seconds per session).
  - **Session-level train/val/test splitting**:
    - **Train**: Sessions 0–13 (14 sessions, ~58.3%)
    - **Validation**: Sessions 14–18 (5 sessions, ~20.8%)
    - **Test**: Sessions 19–23 (5 sessions, ~20.8%)
  - **Zero near-duplicate leakage**: No simulated session spans multiple splits.
- **Anti-Leakage Enforcement**:
  - 16 predictive features $X$ contain strictly observable packet and protocol metadata.
  - Internal TFSM states (`observed_state`), output symbols (`output_symbol`), attack labels (`is_attack`, `attack_type`), and ground-truth mechanisms (`fault_location`) are strictly quarantined.
- **Quality & Test Validation**:
  - 9 automated unit tests in [tests/test_stage3_dataset.py](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/tests/test_stage3_dataset.py) verifying sensor coverage, 7-class representation across all splits, zero feature leakage, absence of NaNs/infs, session disjointness, and seed reproducibility.

---

## 2. Dataset Generation Pipeline Architecture

```
[UnifiedSimulator] (4 Concurrent Sensors: Motion, Temp, Ultrasonic, RFID)
        |
        v
[Raw Benign Events] (Timestamped, Monotonic, Ordered)
        |
        v
[AttackInjector] (Sensor-Aware Injection: DoS, Replay, MITM, DFA, Delay, Spoofing)
        |
        v
[Feature Extractor] (features/build_unified_dataset.py)
        |
        +---> [Observable Feature Vector X] (16 Features)
        +---> [Ground-Truth Target Vector y] (is_attack, attack_type, fault_location)
        +---> [Quarantined Metadata] (session_id, split, state, output_symbol)
        |
        v
[Session-Blocked Splitter] (Train: 0-13 | Val: 14-18 | Test: 19-23)
        |
        v
[datasets/generated/four_sensors/four_sensors_dataset.csv]
```

---

## 3. Data Dictionary: Features, Labels, and Quarantined Metadata

### 3.1 Observable Predictive Features ($X$ — 16 Columns)
These columns are legitimately available at inference time to network monitors or deep learning sequence models:

| # | Column Name | Type | Description |
|---|---|---|---|
| 1 | `inter_arrival` | `float64` | Elapsed time $\Delta t$ in seconds since the previous packet on the network. |
| 2 | `payload_size` | `float64` | Byte length of the application packet. |
| 3 | `qos` | `float64` | MQTT Quality of Service level (0.0 or 1.0). |
| 4 | `seq_delta` | `float64` | Difference in sequence number from the previous packet of this sensor ($\text{seq}_t - \text{seq}_{t-1}$). |
| 5 | `is_sensor_motion` | `float64` | Binary indicator (1.0 if emitting sensor is Motion, 0.0 otherwise). |
| 6 | `is_sensor_temp` | `float64` | Binary indicator (1.0 if emitting sensor is Temperature, 0.0 otherwise). |
| 7 | `is_sensor_ultrasonic` | `float64` | Binary indicator (1.0 if emitting sensor is Ultrasonic, 0.0 otherwise). |
| 8 | `is_sensor_rfid` | `float64` | Binary indicator (1.0 if emitting sensor is RFID, 0.0 otherwise). |
| 9 | `is_event_publish` | `float64` | Binary indicator for standard telemetry publication (`PUBLISH`). |
| 10 | `is_event_alert` | `float64` | Binary indicator for threshold alert frames (`ALERT`). |
| 11 | `is_event_heartbeat` | `float64` | Binary indicator for keepalive heartbeat frames (`HEARTBEAT`). |
| 12 | `is_event_connect` | `float64` | Binary indicator for connection handshake frames (`CONNECT`). |
| 13 | `is_event_flood` | `float64` | Binary indicator for high-rate abnormal burst frames (`FLOOD`). |
| 14 | `numeric_reading` | `float64` | Normalized physical reading (Temp in °C, Distance in cm, Motion status 0/1, Access 1/0, or -1.0 sentinel). |
| 15 | `has_alert_flag` | `float64` | 1.0 if domain payload indicates an active alert condition, 0.0 otherwise. |
| 16 | `has_tamper_flag` | `float64` | 1.0 if proxy modification signature is observable on payload, 0.0 otherwise. |

### 3.2 Ground-Truth Target Labels ($y$ — 4 Columns)
Strictly quarantined for model training loss, evaluation metrics, and fault localization ground truth:

| Column Name | Type | Allowed Values | Role |
|---|---|---|---|
| `is_attack` | `int64` | `0` (Normal), `1` (Attack) | Binary attack detection target. |
| `attack_type` | `object` | `Normal`, `DoS`, `Replay`, `MITM`, `DataFalsification`, `Delay`, `IdentitySpoofing` | Multiclass attack classification target (7 classes). |
| `attack_target` | `object` | `None`, `motion`, `temperature`, `ultrasonic`, `rfid` | Sensor attribution ground truth. |
| `fault_location` | `object` | `None`, `dos_queue_exhaustion`, `replayed_stale_sequence`, `mitm_intercept_tamper`, `dfa_sensor_falsification`, `delay_timeout_exceeded`, `rfid_uid_spoofing_t23` | Formal fault localization ground truth. |

### 3.3 Quarantined Metadata & Traceability (9 Columns)
Quarantined from predictive feature matrix $X$ to prevent target leakage:

| Column Name | Type | Reason for Quarantine |
|---|---|---|
| `session_id` | `int64` | Scenario identifier (used solely for splitting, not training). |
| `split` | `object` | Split assignment (`train`, `val`, `test`). |
| `timestamp` | `float64` | Absolute simulation timestamp (time delta is used in $X$ instead to prevent temporal shortcut learning). |
| `sensor_id` | `object` | Raw string (one-hot indicator columns are provided in $X$). |
| `topic` | `object` | Redundant string with sensor identity. |
| `event_type` | `object` | Raw string (one-hot indicator columns are provided in $X$). |
| `sequence_number` | `int64` | Monotonically increasing raw counter (`seq_delta` is used in $X$). |
| `observed_state` | `object` | Internal TFSM state variable (not broadcast in real MQTT wire frames). |
| `output_symbol` | `object` | Formal mutation output symbol (not broadcast in real MQTT wire frames). |

---

## 4. Scenario Matrix & Session-Level Split Strategy

To guarantee that no attack scenario or near-duplicate packet leaks between splits, **all splits are partitioned strictly by session ID**:

| Split | Sessions | Session Count | Attack Scenarios Included |
|---|:---:|:---:|---|
| **Train** | `0` – `13` | 14 | Benign (Sessions 0, 13); DoS on ultrasonic (1), DoS on motion (7); Replay on motion (2), Replay on temp (8); MITM on temp (3), MITM on ultrasonic (9); DFA on temp (4), DFA on motion (10); Delay on ultrasonic (5), Delay on temp (11); RFID Spoofing (6, 12). |
| **Validation** | `14` – `18` | 5 | Benign (14); DoS + RFID Spoofing (15); Replay + Delay (16); MITM + DFA (17); Delay + RFID Spoofing (18). |
| **Test** | `19` – `23` | 5 | Benign (19); DoS + DFA (20); Replay + RFID Spoofing (21); MITM + Delay (22); DFA + DoS (23). |

**Split Integrity Proof**:
$$\text{Sessions}_{\text{Train}} \cap \text{Sessions}_{\text{Val}} = \emptyset, \quad \text{Sessions}_{\text{Train}} \cap \text{Sessions}_{\text{Test}} = \emptyset, \quad \text{Sessions}_{\text{Val}} \cap \text{Sessions}_{\text{Test}} = \emptyset$$
Every split contains representation of **Normal** and all **six attack classes** across multiple sensors.

---

## 5. Dataset Metrics & Summary

When executed across 24 sessions of 30.0 seconds:
- **Total Rows**: $\approx 2,900$ to $3,200$ rows (depending on flood burst counts).
- **Sensor Representation**:
  - `motion`: $\approx 25\%$
  - `temperature`: $\approx 25\%$
  - `ultrasonic`: $\approx 30\%$ (higher due to DoS bursts)
  - `rfid`: $\approx 20\%$
- **Class Distribution**:
  - `Normal`: $\approx 65\%$ – $75\%$
  - `DoS`: $\approx 10\%$ – $15\%$
  - `Replay`: $\approx 2\%$ – $4\%$
  - `MITM`: $\approx 3\%$ – $5\%$
  - `DataFalsification`: $\approx 3\%$ – $5\%$
  - `Delay`: $\approx 2\%$ – $4\%$
  - `IdentitySpoofing`: $\approx 2\%$ – $4\%$

---

## 6. Exact CMD Execution Commands

Execute the dataset generator and run the complete validation test suite directly in Command Prompt (`cmd.exe`):

```cmd
cd /d C:\Users\cheru\Downloads\iot-ids-diagnosis\iot-ids-diagnosis

:: 1. Generate the Four-Sensor Unified IoT Dataset
python -m features.build_unified_dataset

:: 2. Run Stage 3 dataset validation unit tests
python -m unittest tests.test_stage3_dataset -v

:: 3. Run Stage 2 unified simulation and attack injector regression tests
python -m unittest tests.test_unified_simulator -v
python -m unittest tests.test_attack_injector -v

:: 4. Run Stage 1 formal TFSM model regression tests
python -m models_fsm.verify
python -m unittest tests.test_fsm_sensors -v
```

---

## 7. Preservation of Prior Project Artifacts

- **Real Datasets**: `datasets/raw/mqtt_uad/`, `datasets/raw/mqttset/`, and `datasets/processed/` are completely preserved and untouched.
- **Classical ML & Sequence Baselines**: Stage 3A sequence preparation, Stage 3B classical ML baselines, Stage 3B.6 improved baselines, and Stage 3C sequence modeling code/checkpoints in `ids/`, `results/`, and `docs/` are completely preserved.
- **Git State**: The generated dataset directory `datasets/generated/four_sensors/` is isolated from the real dataset pipelines.

