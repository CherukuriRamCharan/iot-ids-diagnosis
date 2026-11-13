# Stage 2 Report: Unified Four-Sensor IoT Simulation & Sensor-Aware Attack Injection

> **Project:** *Attack Diagnosis and Localization in IoT Intrusion Detection Using Formal Modeling and Deep Learning*  
> **Simulation Scope:** Software simulation of an integrated four-sensor IoT network (Motion, Temperature, Ultrasonic, RFID) with concurrent MQTT telemetry and configurable sensor-aware attack scenarios.  
> **Hardware Notice:** All four sensors and network interfaces in this stage are **pure software simulations** operating in Python. No physical ESP32 microcontrollers, breadboards, or physical sensor integrations are claimed.

---

## 1. Executive Summary

Stage 2 successfully unifies the four verified sensor TFSM models from Stage 1 into a concurrent, multi-sensor IoT network environment:
1. **Unified Simulator (`simulation/unified_simulator.py`)**:
   - Manages four concurrent sensor runtimes (`motion`, `temperature`, `ultrasonic`, `rfid`), each executing its independent TFSM instance.
   - Synchronizes all sensor events on a shared monotonic virtual clock timeline $t$.
   - Adopts a common, standard MQTT message schema with sensor-specific topics.
   - Deterministic event generation governed by explicit random seeds (`numpy.random.default_rng(seed)`).
2. **Attack Injector (`simulation/attack_injector.py`)**:
   - Implements six distinct sensor-aware attack scenarios: **Denial of Service (DoS)**, **Replay**, **Man-in-the-Middle (MITM)**, **Data Falsification (DFA)**, **Message Delay**, and **RFID Identity Spoofing**.
   - Enforces strict **anti-leakage quarantine**: observable features (`timestamp`, `sensor_id`, `topic`, `payload`, `sequence_number`, `qos`, `payload_size`, `observed_state`, `output_symbol`, `inter_arrival`) are completely partitioned from ground-truth audit metadata (`is_attack`, `attack_type`, `attack_target`, `fault_location`, `original_payload`, `original_timestamp`).
3. **Quality & Test Validation**:
   - 8 unit tests in `tests/test_unified_simulator.py` covering multi-sensor concurrency, timestamp monotonicity, schema compliance, and seed reproducibility.
   - 8 unit tests in `tests/test_attack_injector.py` covering all six attack types, interleaved multi-attack scenarios, and anti-leakage boundary enforcement.
   - 100% preservation of all completed Stage 1, Stage 3A, Stage 3B, Stage 3B.6, and Stage 3C artifacts.

---

## 2. System Architecture & Message Schema

### 2.1 Multi-Sensor Concurrency & Isolation
Each sensor maintains an isolated `SensorRuntime` instance containing its formal TFSM, sequence counter, current state, and internal clock. A state transition in one sensor cannot mutate or leak into the state of any other sensor.

```
                          [Shared Virtual Monotonic Timeline]
                                       |
    +-------------------+--------------+--------+-------------------+
    |                   |                       |                   |
[Motion Sensor] [Temperature Sensor]   [Ultrasonic Sensor]  [RFID Access Reader]
topic: .../motion topic: .../temperature topic: distance/... topic: security/rfid1
States: s1..s9  States: s1..s9         States: s1..s9       States: s1..s9
    |                   |                       |                   |
    +-------------------+--------------+--------+-------------------+
                                       |
                          [Common MQTT Event Stream]
                                       |
                     [Sensor-Aware Attack Injector]
                                       |
                     [Interleaved, Monotonic Event Log]
```

### 2.2 Common MQTT Message Schema & Field Partitioning

| Partition | Field Name | Data Type | Description |
|---|---|---|---|
| **Observable** | `timestamp` | `float` | Monotonic virtual simulation time in seconds. |
| **Observable** | `sensor_id` | `str` | Sensor channel identifier (`motion`, `temperature`, `ultrasonic`, `rfid`). |
| **Observable** | `topic` | `str` | MQTT publication topic (e.g. `distance/ultrasonic1`, `home/sensor/motion1`). |
| **Observable** | `event_type` | `str` | Message category (`CONNECT`, `PUBLISH`, `ALERT`, `HEARTBEAT`, `FLOOD`). |
| **Observable** | `payload` | `dict` | Physical reading or device telemetry dictionary. |
| **Observable** | `sequence_number` | `int` | Monotonic per-sensor packet counter. |
| **Observable** | `qos` | `int` | MQTT Quality of Service level (0 or 1). |
| **Observable** | `payload_size` | `int` | Simulated frame byte size. |
| **Observable** | `observed_state` | `str` | Current TFSM state of the emitting sensor. |
| **Observable** | `output_symbol` | `str` | Formal TFSM output symbol emitted (`o2`, `o4`, `o5`, `o11`, `o12`, `o6`). |
| **Observable** | `inter_arrival` | `float` | Elapsed time $\Delta t$ since previous packet on the network stream. |
| **Quarantined** | `is_attack` | `int` | Ground-truth binary target (`0` = Benign, `1` = Attack). **Never in $X$.** |
| **Quarantined** | `attack_type` | `str` | Attack category (`Normal`, `DoS`, `Replay`, `MITM`, `DataFalsification`, `Delay`, `IdentitySpoofing`). |
| **Quarantined** | `attack_target` | `str` | Target sensor subjected to attack. |
| **Quarantined** | `fault_location` | `str` | Attributed formal mutant, transition ID, or exploit mechanism. |
| **Quarantined** | `original_payload` | `dict/None` | Untampered pre-attack payload for validation and audit. |
| **Quarantined** | `original_timestamp`| `float/None`| Original emission time prior to delay injection. |

---

## 3. Attack Injection Mechanisms & Observable Effects

| Attack Scenario | Target Sensor(s) | Injection Mechanism | Observable Physical & Protocol Effect | Ground-Truth Fault Label |
|---|---|---|---|---|
| **1. Denial of Service (DoS)** | Any (e.g. `ultrasonic`) | High-frequency packet flood ($20$–$50$ pkts/sec) during $[t_{\text{start}}, t_{\text{end}}]$. | Extreme inter-arrival drop ($\Delta t \to 0$), event type `FLOOD`, error responses (`o6`). | `dos_queue_exhaustion` |
| **2. Replay Attack** | Any (e.g. `motion`) | Buffers previous legitimate packets and re-injects them after a delay offset. | Non-monotonic / stale sequence numbers (`seq <= last_seq`), duplicate telemetry, error `o6` on replayed connect. | `replayed_stale_sequence` |
| **3. Man-in-the-Middle (MITM)** | Any (e.g. `temperature`) | Intercepts packets in flight, tampers with readings/flags, and adds relay latency. | Inverted alert flags, modified numeric values, increased packet latency. | `mitm_intercept_tamper` |
| **4. Data Falsification (DFA)** | Any (e.g. `temperature`) | Overrides sensor reading with physically impossible values ($T = 185.5^\circ\text{C}$). | Severe out-of-bounds anomaly violating sensor physical constraints; output error code `o6`. | `dfa_sensor_falsification` |
| **5. Message Delay** | Any (e.g. `ultrasonic`) | Delays message delivery by $\Delta t_{\text{delay}} > 2.0\text{s}$ exceeding TFSM keepalive timeout. | Timing constraint violation ($\Delta t > t_{\text{timeout}}$), missing sequence heartbeat, delayed packet arrival. | `delay_timeout_exceeded` |
| **6. RFID Identity Spoofing** | `rfid` | Simulates formal mutant `SPOOF_t23`: unauthorized card UID accepted as authorized. | Forged card UID (`SPOOFED_UNAUTH_UID`), status reported as `authorized`, output symbol `o11` instead of `o12`. | `rfid_uid_spoofing_t23` |

---

## 4. Example Event Records

### A. Benign Telemetry Event (Temperature Sensor)
```python
{
    # --- Observable Fields ---
    "timestamp": 4.5120,
    "sensor_id": "temperature",
    "topic": "home/sensor/temperature1",
    "event_type": "PUBLISH",
    "payload": {"temperature": 23.45, "unit": "Celsius", "humidity": 51.2, "alert": False},
    "sequence_number": 5,
    "qos": 0,
    "payload_size": 104,
    "observed_state": "s9",
    "output_symbol": "o11",
    "inter_arrival": 0.4210,
    # --- Quarantined Metadata ---
    "is_attack": 0,
    "attack_type": "Normal",
    "attack_target": "None",
    "fault_location": "None"
}
```

### B. Injected Data Falsification Event (Temperature Sensor)
```python
{
    # --- Observable Fields ---
    "timestamp": 16.2410,
    "sensor_id": "temperature",
    "topic": "home/sensor/temperature1",
    "event_type": "PUBLISH",
    "payload": {"temperature": 185.5, "alert": True},
    "sequence_number": 14,
    "qos": 0,
    "payload_size": 68,
    "observed_state": "s9",
    "output_symbol": "o6",
    "inter_arrival": 0.5120,
    # --- Quarantined Metadata ---
    "is_attack": 1,
    "attack_type": "DataFalsification",
    "attack_target": "temperature",
    "fault_location": "dfa_sensor_falsification",
    "original_payload": {"temperature": 22.8, "unit": "Celsius", "humidity": 48.0, "alert": False}
}
```

---

## 5. Verification & Test Suite Summary

Stage 2 test execution verified the following test suites:

| Test Suite File | Focus Area | Test Count | Status |
|---|---|:---:|:---:|
| `tests/test_fsm_sensors.py` | Stage 1 Formal TFSM regression (59/59 mutants killed) | 13 | **PASS** |
| `tests/test_unified_simulator.py` | Multi-sensor concurrency, schema, monotonicity, seed reproducibility | 8 | **PASS** |
| `tests/test_attack_injector.py` | All 6 attack types, anti-leakage isolation, interleaved multi-attack scenarios | 8 | **PASS** |
| **Total** | — | **29** | **ALL PASS** |

---

## 6. Exact CMD Execution Commands

To run all Stage 1 and Stage 2 test suites directly in Command Prompt (`cmd.exe`):

```cmd
cd /d C:\Users\cheru\Downloads\iot-ids-diagnosis\iot-ids-diagnosis

:: 1. Verify formal mutation kill rates across all 4 sensors
python -m models_fsm.verify

:: 2. Run Stage 1 formal TFSM unit tests
python -m unittest tests.test_fsm_sensors -v

:: 3. Run Stage 2 unified simulator unit tests
python -m unittest tests.test_unified_simulator -v

:: 4. Run Stage 2 attack injector unit tests
python -m unittest tests.test_attack_injector -v
```

---

## 7. Preservation of Prior Artifacts & Datasets

- **Real Datasets**: `datasets/raw/mqtt_uad/`, `datasets/raw/mqttset/`, and `datasets/processed/` remain untouched.
- **Completed Baselines & Sequences**: Stage 3A, 3B, 3B.6, and 3C scripts, checkpoints, and reports in `ids/`, `results/`, and `docs/` remain 100% intact.
- **Git State**: No large simulated datasets or binary checkpoint files were committed or modified.

