# Paper-to-Project Traceability Matrix
# Attack Diagnosis and Localization in IoT Intrusion Detection

*Project: IoT-Attack-Diagnosis-Localization*
*Reference: Kaddoura et al. (2026) — "Deep learning-based intrusion detection for IoT"*
*Last updated: 2026-10-17*

---

## 1. Overall Paper-to-Project Status

| Paper Section | Paper Component | Implementation Status | Project File(s) |
|---|---|---|---|
| §3 | EFSM for temperature sensor | ✅ Implemented & verified | `models_fsm/temperature_efsm.py`, `temperature_model.md` |
| §3 | EFSM for motion sensor | ✅ Implemented | `models_fsm/motion_efsm.py`, `motion_model.md` |
| §3 | EFSM for ultrasonic motion sensor | ❌ Not implemented (scope limit) | — |
| §3 | EFSM for RFID reader | ❌ Not implemented (scope limit) | — |
| §4 | Attack mutant: MITM | ⚠️ Application-level only | `simulation/attack_scenarios/mitm_simulation.py` |
| §4 | Attack mutant: Replay | ⚠️ Payload-level only | `simulation/attack_scenarios/replay_attack.py` |
| §4 | Attack mutant: Data falsification | ✅ Implemented | `simulation/attack_scenarios/data_falsification.py` |
| §4 | Attack mutant: Battery draining | ⚠️ High-frequency publish only | `simulation/attack_scenarios/battery_draining_simulation.py` |
| §4 | Mutation testing — test suite derivation | ⚠️ Scenarios exist; no formal mutation score | `docs/attack_scenarios/attack_scenario_log.csv` |
| §5 | Network traffic collection | ✅ Done | `captures/`, `data/base_paper/Packets/` |
| §5 | Feature: packet inter-arrival time (IAT) | ✅ Extraction script written | `data/base_paper/add_iat_features.py` |
| §5 | Feature: payload size | ⚠️ Frame length extracted; MQTT payload bytes separate | `features/extract_mqtt_traffic.py` |
| §6 | Deep learning: LSTM | ✅ Implemented | `ids/train_lstm.py` |
| §6 | Deep learning: Transformer | ✅ Implemented | `ids/train_transformer.py` |
| §7 | Baseline comparison (RF vs DL) | ✅ Pipeline ready | `ids/train_rf_5class.py`, `train_lstm.py`, `train_transformer.py` |
| §7 | Per-class metrics + confusion matrices | ✅ Saved to JSON | `results/*_evaluation.json` |
| §7 | Attack diagnosis record | ✅ Implemented | `ids/create_combined_diagnosis.py`, `run_combined_diagnosis.py` |
| §7 | Localization evidence | ✅ Endpoint-level only | `ids/run_combined_diagnosis.py` |

---

## 2. Attack-to-EFSM-to-Feature-to-Detection Traceability

| Attack Scenario | Scenario ID | EFSM State Violated | EFSM Transition | Traffic Feature | Rule/Model Signal | Detection File |
|---|---|---|---|---|---|---|
| Replay-like | ATK-REPLAY-001 | PUBLISH_DATA | repeated_payload_detected | repeated `payload_text`, low IAT variance | `repeated_payload_evidence` (evidence only) | `ids/evaluate_rules_on_captures.py` |
| Data falsification | ATK-FALSIFY-001 | PUBLISH_DATA | invalid_or_unexpected_value | temperature ∉ [−10, 60] °C | `temperature_range_violation` | `ids/evaluate_rules_on_captures.py` |
| Controlled message manipulation | ATK-MANIP-001 | PUBLISH_DATA | message_value_changed | original ≠ forwarded payload on paired topics | `paired_original_forwarded_value_change` | `ids/evaluate_rules_on_captures.py` |
| Excessive communication (battery) | ATK-BATTERY-001 | WAIT_TIMER | excessive_publish_rate | publish count > 20 in 10 s trailing window | `excessive_publish_rate` | `ids/evaluate_packet_rates.py` |
| DoS (SYN flooding) | — (external dataset) | N/A | N/A | high TCP SYN proportion, high pkt/s | ML prediction: `dos` | `ids/train_lstm.py`, `train_transformer.py` |
| Mirai botnet | — (external dataset) | N/A | N/A | packet length distribution, IAT spikes | ML prediction: `mirai` | `ids/train_lstm.py`, `train_transformer.py` |
| MITM (ARP spoofing) | — (external dataset) | N/A | N/A | ARP/IP patterns, low IAT | ML prediction: `mitm` | `ids/train_lstm.py`, `train_transformer.py` |
| Port/host scan | — (external dataset) | N/A | N/A | many destinations, low payload, RST flags | ML prediction: `scan` | `ids/train_lstm.py`, `train_transformer.py` |

---

## 3. Mutation Testing Catalogue

### 3.1 Temperature Sensor Mutants

| Mutant ID | Parent Behaviour | Mutation Applied | Expected EFSM Response | Observed | Capture File |
|---|---|---|---|---|---|
| MUT-TEMP-01 | Normal periodic publish | Replace current payload with previous (replay) | `repeated_payload_detected` → ANOMALOUS | Detected (5 repeated payloads) | `replay_temperature_001.pcapng` |
| MUT-TEMP-02 | Normal periodic publish | Inject out-of-range temperature (95 °C) | `invalid_or_unexpected_value` → ANOMALOUS | Detected (5 out-of-range values) | `data_falsification_temperature_001.pcapng` |
| MUT-TEMP-03 | Normal periodic publish | Forward message with +15 °C offset | `message_value_changed` → ANOMALOUS | Detected (5 modified pairs) | `mitm_simulation.pcapng` |
| MUT-TEMP-04 | Normal periodic publish | Publish at 100× normal rate | `excessive_publish_rate` → ANOMALOUS | Detected (100 PUBLISH packets in burst) | `battery_draining_temperature_001.pcapng` |

### 3.2 Motion Sensor Mutants (defined; simulations pending)

| Mutant ID | Parent Behaviour | Mutation Applied | Expected EFSM Response |
|---|---|---|---|
| MUT-MOTION-01 | Binary 0/1 publish | Repeat last value | `repeated_motion_value` (evidence) |
| MUT-MOTION-02 | Binary 0/1 publish | Inject value 2 (invalid) | `data_falsification` → ANOMALOUS |
| MUT-MOTION-03 | Binary 0/1 publish | Forward with flipped bit | `message_manipulation` → ANOMALOUS |
| MUT-MOTION-04 | Binary 0/1 publish | High-frequency burst | `excessive_communication` → ANOMALOUS |

> Note: MUT-MOTION simulations have not yet been run. Captures and verified detection counts are pending.

---

## 4. Component Implementation Checklist

### ✅ Complete

- [x] Temperature sensor EFSM (states, transitions, 4 detectors)
- [x] Motion sensor EFSM (states, transitions, 4 detectors)
- [x] Temperature attack simulations (4 scenarios, captures saved)
- [x] Normal MQTT captures (2 files)
- [x] External dataset (42 PCAPs, 2.98 M packets, 5 attack classes)
- [x] Dataset feature CSV (`all_packet_features.csv`, 407 MB)
- [x] MQTT feature extraction (TShark-based, CSV output)
- [x] MQTT payload extraction and decoding
- [x] Random Forest binary classifier (trained, saved)
- [x] Random Forest 5-class script (written, requires IAT CSV first)
- [x] IAT feature enrichment script (`add_iat_features.py`)
- [x] LSTM intrusion detection (full training pipeline)
- [x] Transformer intrusion detection (full training pipeline)
- [x] Payload rule evaluation (range, manipulation, repeated)
- [x] Packet rate evaluation
- [x] Combined diagnosis report (per-capture)
- [x] Capture-level prediction and evaluation
- [x] Unit tests: diagnosis rules (7 tests)
- [x] Unit tests: EFSM transitions, schema, model loading, sequences (extended)
- [x] Path-bug fixes in all `ids/` scripts
- [x] `final_evaluation.py` column mismatch fixed
- [x] Paper-to-project traceability matrix (this file)
- [x] Mutation testing catalogue

### ⚠️ Partially Complete

- [ ] IAT CSV generated (script written; **must be run** before LSTM/Transformer)
- [ ] 5-class RF trained (script written; **must be run**)
- [ ] LSTM trained (script written; **must be run**)
- [ ] Transformer trained (script written; **must be run**)
- [ ] Motion sensor attack simulations (EFSM done; captures not yet collected)
- [ ] Architecture diagram (described in README; no image file yet)

### ❌ Out of Scope (documented)

- [ ] Ultrasonic motion sensor EFSM
- [ ] RFID reader EFSM
- [ ] Formal mutation score calculation
- [ ] Physical device localization (endpoint-level only — by design)
- [ ] Attacker attribution

---

## 5. Evaluation Honest Summary

| Model | Task | Split | Accuracy | Benign F1 | Notes |
|---|---|---|---|---|---|
| RF binary | Packet-level, 2 classes | Capture-grouped attack; in-capture temporal benign | 96.1% | 0.58 | Benign test is same-capture temporal proxy |
| RF 5-class | Packet-level, 5 classes | Same as above | *Run train_rf_5class.py* | — | IAT features optional |
| LSTM | Sequence-level, 5 classes | Capture-grouped 70/15/15 | *Run train_lstm.py* | — | 20-packet windows |
| Transformer | Sequence-level, 5 classes | Same as LSTM | *Run train_transformer.py* | — | Pre-norm encoder |
| Rule-based | Capture-level, 4 signals | All 6 local captures | 4/4 simulated attacks detected | — | Calibrated on 1 normal capture only |
| Capture-level RF | Capture-level, 2 classes | Same model, 42 captures | 100% | — | NOT independent validation; 41/42 captures are attack |
