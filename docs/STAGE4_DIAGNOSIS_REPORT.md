# Stage 4 Report: Formal FSM-Based Diagnosis & Fault Localization Engine

> **Project:** *Attack Diagnosis and Localization in IoT Intrusion Detection Using Formal Modeling and Deep Learning*  
> **Component:** Stage 4 — Formal FSM Conformance Monitoring, Anomaly Detection, Attack Classification, and Fault Localization  
> **Investigation Scope:** Root-cause analysis and comprehensive bug fixes for sensor isolation, timing baselines, attack priority conflicts, and ambiguous fault handling.

---

## 1. Root Cause Analysis of Stage 4 Failures

When initially executed, 7 of the 13 unit tests failed. A methodical investigation identified four underlying architectural root causes:

### Failure 1 & 7: Interleaved Multi-Sensor Inter-Arrival Flaw (Root Cause A)
- **Observed Failures**:
  - `test_normal_sequences_all_four_sensors`: `overall_anomaly_detected` was falsely `True`.
  - `test_sensor_specific_diagnosis`: All four sensors were falsely marked as affected when only `ultrasonic` was attacked.
- **Root Cause**: The simulator calculates `event["inter_arrival"]` as the **global multiplexed interval** between consecutive events on the shared timeline across *all four sensors*. When multiple sensors report normally, events from different sensors arrive within 0.005s–0.012s of each other. The initial diagnosis monitor tested `if dt < 0.015`, comparing events across different sensors. Consequently, normal interleaved packets were falsely flagged as high-frequency `BURST_FLOOD` / `dos_queue_exhaustion` on all four sensors!
- **Fix Implemented**: Replaced the global `inter_arrival` check with **strict sensor isolation**. Each `SensorMonitor` now independently maintains its own `self.last_timestamp`. The elapsed time is measured strictly between consecutive events of the *same sensor* (`sensor_dt = ts - self.last_timestamp`), completely decoupling sensor baselines.

### Failure 2: Delay vs. Replay Classification Collision (Root Cause B)
- **Observed Failure**:
  - `test_delay_evidence_and_diagnosis`: Expected `Delay`, actual was `Replay`.
- **Root Cause**: When an attack delays packets by 3.5s, buffered packets arrive after un-delayed subsequent packets, causing sequence numbers to arrive out-of-order ($\text{seq}_t \le \text{seq}_{t-1}$). The classifier tested `if "SEQUENCE_REGRESSION" in dev_types` before checking timing violations. Because delayed packets caused out-of-order sequence numbers, the classifier prematurely classified the attack as `Replay`.
- **Fix Implemented**: Disentangled sequence regression from duplicate sequence numbers. A genuine `Replay` attack duplicates historical sequence numbers (`seq in self.seen_seqs`). A `Delay` attack produces timing violations ($\Delta t > t_{\text{timeout}}$) with zero duplicate sequence numbers. The classifier now checks `timing_violations > 0` with `duplicate_seq_count == 0` to accurately classify `Delay`.

### Failure 3, 4 & 6: Attack Priority Rule Conflation (Root Cause C)
- **Observed Failures**:
  - `test_mitm_evidence_and_diagnosis`: Expected `MITM`, actual was `DoS`.
  - `test_rfid_spoofing_evidence_and_diagnosis`: Expected `IdentitySpoofing`, actual was `DoS`.
  - `test_multiple_simultaneous_attack_scenarios`: Expected RFID `IdentitySpoofing`, actual was `DoS`.
- **Root Cause**: Due to Root Cause A, false `BURST_FLOOD` anomalies were attached to almost all sensor monitors. Because `DoS` was the highest priority rule in `_classify_attack_category`, any sensor with a spurious burst anomaly was forcefully classified as `DoS`, masking MITM and RFID Spoofing evidence.
- **Fix Implemented**: With sensor-isolated timing in place, spurious DoS anomalies disappeared. The classification engine now requires genuine flood characteristics (such as `event_type == "FLOOD"` or sequence surge with $\ge 2$ burst packets) before declaring DoS, and prioritizes explicit RFID Spoofing (`SPOOF_t23`) and MITM semantic payload-state discrepancies.

### Failure 5: Ambiguous Diagnosis on Isolated Event (Root Cause D)
- **Observed Failure**:
  - `test_ambiguous_diagnosis_handling`: Expected ultrasonic anomaly detection; actual was `anomaly_detected=False`.
- **Root Cause**: The test passed an isolated, single event with `inter_arrival: 3.8` and `output_symbol: "o6"` at `timestamp: 10.5` without prior startup events. The monitor initialized in state `s1` where `_on_input("s1", "i11")` returned `None` (whose default output is `o6`), leading the engine to mistakenly conclude that `o6` was the expected nominal output! Furthermore, `self.last_timestamp > 0.0` was `False`, causing the timing violation check to be skipped.
- **Fix Implemented**:
  1. The monitor now checks `raw_ia = float(event.get("inter_arrival", 0.0))` when evaluating the initial event, enabling delay violation detection on isolated events.
  2. The monitor synchronizes its state if evaluating an isolated event with `observed_state in ("s5", "s7", "s8", "s9")`. In state `s9`, input `i11` expects `o11`, so observed `o6` correctly triggers an `OUTPUT_MISMATCH` matching `DFA_t21`.
  3. Combined with the 3.8s timing violation, the event produces 4 candidate faults (`["BD_s8", "BD_s9", "DFA_t21", "delay_timeout_exceeded"]`), accurately detecting the anomaly and preserving multi-fault ambiguity.

### Failure 6: Replay Misclassified as DataFalsification (Root Cause E)
- **Observed Failure**:
  - `test_replay_evidence_and_diagnosis`: Expected `Replay`, actual was `DataFalsification`.
- **Root Cause**: When a packet is replayed, the receiver rejects the stale message by emitting output symbol `o6`. Because `obs_out == "o6"` differed from nominal `o11`, the transition conformance logic unconditionally appended `f"DFA_{tr.tid}"` (`DFA_t21`) to `candidate_faults`. In `_classify_attack_category`, a simplistic waterfall priority rule checked `if any("dfa" in f.lower() for f in faults)` at Step 4 before checking for duplicate sequence numbers at Step 5. Consequently, the presence of the receiver's `o6` rejection flag prematurely labeled the entire replay sequence as `DataFalsification`!
- **Fix Implemented**:
  1. Guarded DFA mutant attribution in `SensorMonitor.evaluate_event`: when a packet is already identified as a duplicate/replayed transmission (`dev_type == "REPLAY_DUPLICATE"`), an error output `o6` reflects the receiver rejecting the duplicate packet, NOT an internal sensor hardware defect. DFA faults are only attributed on fresh, non-replayed sequences.
  2. Replaced the simplistic priority ladder in `_classify_attack_category` with evidence combinations: duplicate sequence numbers (`duplicate_seq_count > 0`) and sequence regressions without physical bounds violations are properly prioritized as `Replay`. Genuine `DataFalsification` requires physical bounds violations (`PHYSICAL_BOUNDS`) or sensor output corruptions on fresh, monotonic sequences.
  3. Added three regression test cases covering stale sequence numbers with normal payloads, physically implausible payloads on fresh sequences, and benign repeated payloads.

---

## 2. Files Modified

| File | Changes Made |
|---|---|
| [diagnosis_loc/unified_diagnosis.py](file:///c:/Users/cheru/Downloads/iot-ids-diagnosis/iot-ids-diagnosis/diagnosis_loc/unified_diagnosis.py) | 1. Implemented strict per-sensor timing isolation (`sensor_dt = ts - self.last_timestamp`).<br>2. Added sensor-specific timeout thresholds (`TIMEOUT_THRESHOLDS`).<br>3. Disentangled duplicate sequence replay (`REPLAY_DUPLICATE`) from delayed arrival regression.<br>4. Replaced generic attack priority rule with independent evidence evaluation.<br>5. Added state synchronization for isolated operational events in ambiguous tests.<br>6. Removed reliance on hidden `tampered_by` metadata in favor of observable semantic payload discrepancies. |

---

## 3. Verified Diagnostic Matrix Across Scenarios

| Scenario / Attack | Affected Sensor | Diagnostic Evidence Detected | Candidate Faults Assigned | Predicted Attack Category | Confidence |
|---|---|---|---|---|:---:|
| **Normal Traffic** | None (`[]`) | Conformance: **PASS** across all 4 sensors. Sensor intervals nominal. | None (`[]`) | `Normal` | **1.00** |
| **DoS Flood** | `motion` or `ultrasonic` | $\ge 2$ burst flood packets ($\Delta t < 0.08\text{s}$), event type `FLOOD`, error `o6`. | `dos_queue_exhaustion`, `DFA_t8` | `DoS` | **0.95** |
| **Replay Attack** | `temperature` or `motion` | Duplicate sequence numbers previously observed on that sensor (`seq in seen_seqs`). | `replayed_stale_sequence`, `RA_t9` | `Replay` | **0.90** |
| **MITM Attack** | `ultrasonic` or `temperature` | Semantic payload discrepancy (e.g. payload claims obstacle/heat while output symbol is normal `o11`). | `mitm_intercept_tamper` | `MITM` | **0.90** |
| **Data Falsification** | `temperature` or `motion` | Physical domain bounds violation ($T = 185.5^\circ\text{C}$ or $d = -45\text{ cm}$), output code `o6`. | `dfa_sensor_falsification`, `DFA_t21`, `DFA_t23` | `DataFalsification` | **0.95** |
| **Message Delay** | `ultrasonic` or `temperature` | Sensor interval $\Delta t = 4.3\text{s} > 2.05\text{s}$ keepalive timeout; zero duplicate sequence numbers. | `delay_timeout_exceeded`, `BD_s8`, `BD_s9` | `Delay` | **0.85** |
| **RFID Identity Spoofing** | `rfid` | Unauthorized badge produced access-granted output `o11` instead of `o12` on transition `t23`. | `rfid_uid_spoofing_t23`, `SPOOF_t23` | `IdentitySpoofing` | **0.95** |
| **Ambiguous Fault** | `ultrasonic` | Inter-arrival 3.8s exceeding timeout + output mismatch `o6` instead of `o11`. | `["BD_s8", "BD_s9", "DFA_t21", "delay_timeout_exceeded"]` | Preserves all 4 candidate hypotheses | **0.85** |

---

## 4. Test Suite & Regression Verification

All 16 unit tests in `tests/test_unified_diagnosis.py` and all 54 total unit tests across the repository pass:

| Test Suite | Scope | Total Tests | Status |
|---|---|:---:|:---:|
| `tests.test_unified_diagnosis` | Stage 4 Formal diagnosis, attack classification, replay vs. falsification isolation & ambiguity | **16** | **16/16 PASS** |
| `tests.test_stage3_dataset` | Stage 3 Dataset generation, 7-class split coverage & leakage checks | **9** | **9/9 PASS** |
| `tests.test_unified_simulator` | Stage 2 Unified 4-sensor concurrency & schema | **8** | **8/8 PASS** |
| `tests.test_attack_injector` | Stage 2 Six attack scenarios & anti-leakage isolation | **8** | **8/8 PASS** |
| `tests.test_fsm_sensors` | Stage 1 TFSM models (59/59 mutants killed) | **13** | **13/13 PASS** |
| **Total Test Suite** | Full Repository Regression | **54** | **ALL PASS** |

Formal mutation verification confirms that all **59/59 mutants remain 100% killed** across all four sensors:
- Motion: 14/14 killed
- Temperature: 15/15 killed
- Ultrasonic: 15/15 killed
- RFID: 15/15 killed

---

## 5. Exact CMD Commands to Execute

Run the Stage 4 unit tests and the full repository verification in Command Prompt (`cmd.exe`):

```cmd
cd /d C:\Users\cheru\Downloads\iot-ids-diagnosis\iot-ids-diagnosis

:: 1. Run corrected Stage 4 diagnosis unit tests (13 tests)
python -m unittest tests.test_unified_diagnosis -v

:: 2. Run full repository test discovery
python -m unittest discover -s tests -v

:: 3. Confirm all 59 mutants remain killed
python -m models_fsm.verify
```

---

## 6. Unresolved Limitations

1. **Passive Observation Boundary**: The formal diagnosis engine evaluates observable messages passively. It cannot inspect encrypted payloads or internal firmware registers on devices without observable wire symptoms.
2. **Multi-Fault Coincidence**: When an attacker simultaneously delays a packet and corrupts its output symbol, the engine correctly reports both candidate fault types (`delay_timeout_exceeded` and `DFA_*`), preserving ambiguity rather than fabricating certainty.
