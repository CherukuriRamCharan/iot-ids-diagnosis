# Dataset Selection Justification & Multi-Track Strategy

> **Project:** Attack Diagnosis and Localization in IoT Intrusion Detection Using Formal Modeling and Deep Learning  
> **Ground Truth Basis:** Verified findings from `docs/DEEP_DATASET_ANALYSIS.md`, `docs/MQTT_UAD_DATASET_AUDIT.md`, `docs/MQTTSET_DATASET_AUDIT.md`, and `README.md`.

---

## 1. Why MQTT-UAD and MQTTset Cannot Be Merged

Merging `MQTT-UAD` and `MQTTset` into a single unified table is **scientifically and technically invalid** for five verified reasons:

1. **Incompatible Granularity & Schemas**:
   - `MQTT-UAD` is **packet-level** (67 Wireshark/TShark columns: `frame.time_delta`, `eth.src`, `ip.src`, `mqtt.msgtype`, `mqtt.topic`, `mqtt.msg`, `mqtt.qos`, etc.).
   - `MQTTset` is **flow-level** (85 CICFlowMeter-style columns: `Flow_Duration`, `Tot_Fwd_Pkts`, `Flow_IAT_Mean`, `Pkt_Len_Std`, etc.) with **zero** MQTT application-layer fields.
2. **Disjoint Capture Epochs**:
   - `MQTT-UAD` was captured in 2018 (`1522235370`–`1535977010`, March–September 2018).
   - `MQTTset` was captured across 2019–2021 (`24/07/2019`, `10/06/2020`–`01/07/2020`, `15/07/2021`).
3. **Incompatible Label Taxonomies**:
   - `MQTT-UAD` uses `type` ∈ {`normal`, `DoS`, `mitm`, `intrusion`}.
   - `MQTTset` uses `Cat` ∈ {`Normal`, `Flood`, `MalariaDoS`, `Bruteforce`, `Malformed`, `Slowite`}.
4. **Incompatible Normal-Traffic Baselines**:
   - `MQTT-UAD` `normal` packets are real lab capture frames (`eth.*`, `ip.*`, `tcp.*`) where 97%+ are background non-MQTT traffic.
   - `MQTTset` `Normal` flows have `Protocol=0`, `Dst_Port=0`, and 83,283 byte-swapped `Src_IP`s.
5. **Cross-Domain Leakage Risk**: Concatenating them would let a classifier trivially separate datasets by feature availability or null patterns rather than attack semantics.

---

## 2. Recommended Multi-Track Dataset Strategy

Because your project (`README.md`) already uses **TFSM + mutation testing (`models_fsm/`, `simulation/`)** for formal fault localization and wants real/external datasets for **MQTT-specific** and **network-flow** validation, the strongest defensible design is a **three-track complementary strategy**:

| Track | Data Source | Role in Project | What It Evaluates |
|---|---|---|---|
| **Track A (Primary Formal Diagnosis + Localization)** | Existing `models_fsm/` + `simulation/` TFSM mutant traffic | **Attack Diagnosis + Exact Fault Localization** | Two-head LSTM predicting attack class (`DFA`, `MITM`, `BD`, `Replay`) **and** faulty TFSM transition/state (mutant ID), with session-wise splits. |
| **Track B (Packet/Protocol-Level MQTT Validation)** | **MQTT-UAD** (`DoS.csv`, `MitM.csv`, `Intrusion.csv`) | **Real Packet & MQTT-Layer Detection / Diagnosis** | Evaluates detection & 4-class classification (`normal`, `DoS`, `mitm`, `intrusion`) on real Wireshark packet captures using timing (`frame.time_delta`), size (`frame.len`), and MQTT header/protocol fields (`mqtt.msgtype`, `mqtt.len`, `mqtt.qos`, `mqtt.dupflag`, `mqtt.retain`). |
| **Track C (Flow-Level Statistical Validation — Secondary/Caveated)** | **MQTTset** (`MQTTset.csv`) | **Flow-Level Multi-Class Anomaly Classification** | Evaluates 6-class flow-level classification (`Normal`, `Flood`, `MalariaDoS`, `Bruteforce`, `Malformed`, `Slowite`) **strictly after** removing identifier/leakage columns (`Flow_ID`, `Src_IP`, `Dst_IP`, `Src_Port`, `Dst_Port`, `Protocol`, `Timestamp`) and constant/infinite values. |

---

## 3. Justified Preprocessing & Anti-Leakage Rules (Before Any Modeling)

### 3.1 Rules for MQTT-UAD (Track B)
1. **Drop 19–22 all-NaN columns and 4 constant columns** (`frame.encap_type`, `frame.ignored`, `frame.marked`, `frame.offset_shift`).
2. **Drop capture-identifying and shortcut-leakage columns**:
   - `frame.number`, `frame.time_epoch`, `frame.time_relative` (uniquely identify capture file and attack window).
   - `ip.src`, `ip.dst`, `eth.src`, `eth.dst` for generalizable ML training (keep only for post-hoc alert attribution / host-level reporting, not as raw classification features, because `b8:27:eb:73:32:30` and `192.168.1.196` leak single-lab attacker identity).
   - `mqtt.topic` and `mqtt.clientid` as raw strings in `DoS` classification (since `mqtt-malaria/...` appears only in `DoS`).
3. **Use chronological / session-blocked splits** (not random row shuffle): `DoS.csv` has 76,285 near-duplicates and `MitM.csv` has 98,488 near-duplicates when ignoring `frame.number`/`frame.time_*`; random splitting would cause severe train-test leakage.
4. **Address severe class imbalance**: `mitm` (1.35%) and `intrusion` (0.66%) require class-weighted loss or stratified block evaluation (macro-F1, per-class PR-AUC, not accuracy alone).

### 3.2 Rules for MQTTset (Track C)
1. **Mandatory removal of shortcut-leakage columns**: `Flow_ID`, `Src_IP`, `Dst_IP`, `Src_Port`, `Dst_Port`, `Protocol`, `Timestamp`. (Verified: `Protocol=0, Dst_Port=0` alone separates 100% of `Normal` from `Anomaly`, and `Src_IP`/`Dst_IP` separates `Flood`, `Slowite`, and the `.90/.91` group.)
2. **Drop 15 constant columns** (e.g., `Bwd_PSH_Flags`, `Fwd_URG_Flags`, `Active_Mean`, etc.).
3. **Clean `±inf` values**: `Flow_Pkts/s` (328 `±inf`) and `Flow_Byts/s` (193 `±inf`).
4. **Deduplicate on feature columns** (`688` duplicate feature vectors ignoring `Flow_ID` and `Timestamp`) before splitting.
5. **Report MQTTset as a secondary flow-level benchmark with explicit artifact caveats**: Even after dropping `Protocol`/`Dst_Port`, `Normal` flows in `MQTTset.csv` have `Tot_Bwd_Pkts = 0` and `TotLen_Fwd_Pkts = 0` in the inspected sample rows, reflecting a different capture pipeline from the TCP-1883 attack flows.

---

## 4. Summary Recommendation

- **Do not merge** `MQTT-UAD` and `MQTTset`.
- **Use `MQTT-UAD` as the primary external packet/MQTT dataset** for validating detection and protocol-level diagnosis (`normal`, `DoS`, `mitm`, `intrusion`), paired with your **TFSM + mutant simulation** for formal fault localization.
- **Use `MQTTset` only as a secondary flow-level benchmark** (with all IP/port/protocol/timestamp columns removed and capture-bias limitations explicitly disclosed).

