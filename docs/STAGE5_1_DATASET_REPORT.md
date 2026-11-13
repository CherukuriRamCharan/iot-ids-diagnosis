# Stage 5.1 Report: Four-Sensor Dataset & Sequence Pipeline

> **Project:** *Attack Diagnosis and Localization in IoT Intrusion Detection Using Formal Modeling and Deep Learning*  
> **Component:** Stage 5.1 — Four-Sensor Tabular Loading, Feature Provenance Audit, Anti-Leakage Scaler Fitting, and Sequence Window Generation  
> **Module Created:** `ids/datasets_four_sensors.py`  
> **Test Suite Created:** `tests/test_stage5_dataset.py`

---

## 1. Feature Provenance Audit & Final Feature Selection

Every candidate column in `datasets/generated/four_sensors/four_sensors_dataset.csv` was subjected to an operational provenance audit to determine whether it reflects genuine physical or network telemetry available during real-time inference, or represents simulator/injector metadata that must be strictly excluded.

### A. Selected Observable Features (14 Features)

The final feature set comprises **14 strictly observable** network and telemetry fields:

| # | Feature Name | Provenance & Source | Type | Normalization |
|---|---|---|:---:|:---:|
| 1 | `inter_arrival` | Global timeline delta between packet arrivals (seconds). | Continuous | Standardized |
| 2 | `payload_size` | Length of serialized MQTT message in bytes. | Continuous | Standardized |
| 3 | `qos` | MQTT Quality of Service level (0 or 1). | Binary Flag | Preserved [0, 1] |
| 4 | `seq_delta` | Monotonic progression delta ($\text{seq}_t - \text{seq}_{t-1}$) per sensor. | Continuous | Standardized |
| 5 | `is_sensor_motion` | One-hot topic flag: `home/sensor/motion1`. | Binary Flag | Preserved [0, 1] |
| 6 | `is_sensor_temp` | One-hot topic flag: `home/sensor/temperature1`. | Binary Flag | Preserved [0, 1] |
| 7 | `is_sensor_ultrasonic` | One-hot topic flag: `distance/ultrasonic1`. | Binary Flag | Preserved [0, 1] |
| 8 | `is_sensor_rfid` | One-hot topic flag: `security/rfid1`. | Binary Flag | Preserved [0, 1] |
| 9 | `is_event_publish` | Protocol control flag: standard telemetry `PUBLISH`. | Binary Flag | Preserved [0, 1] |
| 10 | `is_event_alert` | Protocol control flag: sensor event `ALERT`. | Binary Flag | Preserved [0, 1] |
| 11 | `is_event_heartbeat` | Protocol control flag: periodic `HEARTBEAT` / PING. | Binary Flag | Preserved [0, 1] |
| 12 | `is_event_connect` | Protocol control flag: initial handshake `CONNECT`. | Binary Flag | Preserved [0, 1] |
| 13 | `numeric_reading` | Parsed sensor payload value ($T$ in $^\circ\text{C}$, $d$ in $\text{cm}$, etc.; $-1.0$ sentinel). | Continuous | Standardized (Masked) |
| 14 | `has_alert_flag` | Application payload boolean (`alert`, `obstacle`, `motion`). | Binary Flag | Preserved [0, 1] |

### B. Explicit Feature Exclusions & Leakage Justifications

| Excluded Field | Origin & Audit Finding | Justification for Exclusion |
|---|---|---|
| **`has_tamper_flag`** | Generated from `payload["tampered_by"] == "mitm_proxy"`. | **Blatant Ground-Truth Leakage.** In `simulation/attack_injector.py`, the MITM attack injector marked tampered packets with `tampered_by = "mitm_proxy"`. Real network packets from an adversary proxy never carry identifying debug watermarks. Excluded to force the model to learn physical/semantic anomalies rather than looking for a simulator cheat code. |
| **`is_event_flood`** | Generated from synthetic `event_type == "FLOOD"`. | **Artificial Classification Shortcut.** In real IoT networks, packet headers do not label DoS bursts as "FLOOD". A DoS flood consists of standard `PUBLISH` packets sent at abnormal rates. Excluded to ensure DoS detection relies strictly on inter-arrival timing, sequence surges, and payload sizes. |
| **`observed_state`** | Internal TFSM runtime state (`s1`..`s9`). | **Quarantined.** Internal state machine states are formal mathematical constructs inside the monitor, not physical wire observables. |
| **`output_symbol`** | Internal TFSM output symbol (`o1`..`o13`). | **Quarantined.** Emitted by the formal monitor during conformance checking; cannot be consumed as an input feature for sequence modeling. |
| **`is_attack`** | Target binary label. | Target variable. |
| **`attack_type`** | Target multiclass label. | Target variable. |
| **`attack_target`** | Injector target metadata. | Simulator ground-truth metadata. |
| **`fault_location`** | Injector mutant metadata. | Formal localization label. |

---

## 2. Event and Sequence Distributions Across Splits

### A. Raw Event Distribution

| Class | Train Events (Sessions 0–13) | Val Events (Sessions 14–18) | Test Events (Sessions 19–23) | Total Events | % of Total |
|---|:---:|:---:|:---:|:---:|:---:|
| **Normal** | 1,567 | 546 | 558 | 2,671 | 86.44% |
| **DoS** | 164 | 51 | 101 | 316 | 10.23% |
| **IdentitySpoofing** | 21 | 14 | 6 | 41 | 1.33% |
| **Delay** | 8 | 7 | 3 | 18 | 0.58% |
| **DataFalsification** | 8 | 3 | 7 | 18 | 0.58% |
| **MITM** | 8 | 2 | 5 | 15 | 0.49% |
| **Replay** | 6 | 3 | 2 | 11 | 0.36% |
| **Split Total** | **1,782** | **626** | **682** | **3,090** | **100.00%** |

### B. Sequence Window Distribution ($L=16, \text{Stride}=1, \text{Label}=\text{Last}$)

Because sequence windows of length $L=16$ are strictly constrained within individual session boundaries:
- A session with $N$ events yields exactly $N - 16 + 1 = N - 15$ sequence windows.
- In Train (14 sessions: 0–13): $1,782 - (14 \times 15) = 1,572$ sequence windows.
- In Val (5 sessions: 14–18): $626 - (5 \times 15) = 551$ sequence windows.
- In Test (5 sessions: 19–23): $682 - (5 \times 15) = 607$ sequence windows.
- Total sequence windows generated: **2,730 sequences**.

Because all attacks are initiated at least $5.0\text{s}$ into each session (after row index $\ge 25$), **no attack event is lost to the initial 15-event warmup window**. Every single attack event in the dataset is preserved as the target label of at least one sequence window:

| Class | Train Sequences | Val Sequences | Test Sequences | Total Sequences | % of Total |
|---|:---:|:---:|:---:|:---:|:---:|
| **Normal** | 1,357 | 471 | 483 | 2,311 | 84.65% |
| **DoS** | 164 | 51 | 101 | 316 | 11.58% |
| **IdentitySpoofing** | 21 | 14 | 6 | 41 | 1.50% |
| **DataFalsification** | 8 | 3 | 7 | 18 | 0.66% |
| **MITM** | 8 | 2 | 5 | 15 | 0.55% |
| **Delay** | 8 | 7 | 3 | 18 | 0.66% |
| **Replay** | 6 | 3 | 2 | 11 | 0.40% |
| **Multiclass Total** | **1,572** | **551** | **607** | **2,730** | **100.00%** |
| *Binary: Normal* | 1,357 (86.32%) | 471 (85.48%) | 483 (79.57%) | 2,311 (84.65%) | — |
| *Binary: Attack* | 215 (13.68%) | 80 (14.52%) | 124 (20.43%) | 419 (15.35%) | — |

---

## 3. Class Imbalance Analysis & Computed Class Weights

### Inverse-Frequency Class Weights
Computed strictly on the training sequence labels ($N_{\text{tr}} = 1,572$, $K=7$):
$$w_c = \frac{N_{\text{samples}}}{K \cdot N_c}$$

| Class ID | Class Name | Training Count ($N_c$) | Computed Class Weight ($w_c$) | Weight Relative to Normal |
|:---:|---|:---:|:---:|:---:|
| 0 | **Normal** | 1,357 | **0.1655** | $1.0\times$ |
| 1 | **DoS** | 164 | **1.3693** | $8.3\times$ |
| 2 | **Replay** | 6 | **37.4286** | $226.2\times$ |
| 3 | **MITM** | 8 | **28.0714** | $169.6\times$ |
| 4 | **DataFalsification** | 8 | **28.0714** | $169.6\times$ |
| 5 | **Delay** | 8 | **28.0714** | $169.6\times$ |
| 6 | **IdentitySpoofing** | 21 | **10.6939** | $64.6\times$ |

Binary weights on training sequence labels ($N_{\text{tr}} = 1,572$, $K=2$):
- **Normal** ($N_0 = 1,357$): $w_0 = 1572 / (2 \times 1357) = \mathbf{0.5792}$
- **Attack** ($N_1 = 215$): $w_1 = 1572 / (2 \times 215) = \mathbf{3.6558}$ ($6.31\times$ weight ratio)

---

## 4. Fundamental Limitations of Minority-Class Scarcity

> [!IMPORTANT]
> **Severe Minority Imbalance Boundary**:
> While inverse-frequency class weighting ensures that recurrent gradient updates do not ignore minority attack instances during loss computation, **class weighting does not synthesize new data**.

1. **Extreme Support Scarcity**:
   - `Replay` has only **6 training sequences** and **2 test sequences**.
   - `MITM` has only **8 training sequences** and **5 test sequences**.
   - `Delay` has only **8 training sequences** and **3 test sequences**.
   - `DataFalsification` has only **8 training sequences** and **7 test sequences**.
2. **Impact on Test Metric Volatility**:
   - For `Replay` (with only 2 test instances), predicting 1 of 2 yields a recall of exactly $0.50$; missing both yields $0.00$. A single false positive drops precision to $\le 0.50$.
   - Macro-F1 across all 7 classes is exceptionally sensitive to single-instance predictions on these sparse classes.
3. **Architectural Justification for the Two-Tier Hybrid IDS**:
   - This acute statistical scarcity is the precise justification for integrating deep learning with a formal Timed Finite State Machine.
   - When a deep learning sequence model lacks statistical sample density to achieve high-confidence pattern recognition on rare attacks like `Replay` or `Delay`, the formal TFSM operates deterministically on transition and timeout invariants ($100\%$ detection of duplicate sequences and timeout breaches) regardless of sample scarcity.

---

## 5. Verification Commands (Windows CMD)

To verify the Stage 5.1 dataset pipeline and unit tests in Command Prompt:

```cmd
cd /d C:\Users\cheru\Downloads\iot-ids-diagnosis\iot-ids-diagnosis

:: 1. Run Stage 5.1 dataset pipeline tests
python -m unittest tests.test_stage5_dataset -v

:: 2. Confirm full repository tests remain passing (65 tests)
python -m unittest discover -s tests -v

:: 3. Confirm all 59 mutants remain killed
python -m models_fsm.verify
```

