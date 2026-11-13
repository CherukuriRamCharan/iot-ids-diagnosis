# Attack Diagnosis and Localization Feasibility Assessment

> **Project:** Attack Diagnosis and Localization in IoT Intrusion Detection Using Formal Modeling and Deep Learning  
> **Evidence Basis:** Verified column schemas, non-null counts, and value distributions from `docs/DEEP_DATASET_ANALYSIS.md`. No fabricated results or unsupported localization claims are made.

---

## 1. Feasibility Matrix: Detection vs. Classification vs. Diagnosis vs. Localization

| Capability | Definition in This Project | MQTT-UAD (`DoS`, `MitM`, `Intrusion`) | MQTTset (`MQTTset.csv`) | Project TFSM + Mutant Simulation (`models_fsm/`, `simulation/`) |
|---|---|---|---|---|
| **1. Attack Detection** (Binary: Normal vs. Attack) | Identify whether traffic is benign or malicious | **Supported** (`normal`: 234,919 vs. attack: 51,267). Caveat: `mitm` (3,855) and `intrusion` (1,898) are small, and `normal` is 97%+ non-MQTT traffic. | **Technically possible, heavily caveated** (`Normal`: 440,699 vs. `Anomaly`: 100,372). `Protocol=0, Dst_Port=0` in `Normal` creates severe capture-artifact bias. | **Supported** (binary collapse of mutant vs. nominal traces). |
| **2. Attack Classification** (Multi-class Attack Family) | Distinguish attack types (`DoS`, `mitm`, `intrusion`, etc.) | **Supported** (4 classes via `type`: `normal`, `DoS`, `mitm`, `intrusion`). Caveat: each attack class comes from a separate capture file/date. | **Partially Supported** (6 classes via `Cat`: `Normal`, `Flood`, `MalariaDoS`, `Bruteforce`, `Malformed`, `Slowite`), **only if** leakage columns (`Protocol`, `Dst_Port`, `Src_IP`, `Dst_IP`, `Timestamp`, `Flow_ID`) are removed. | **Supported** (`DFA`, `MITM`, `BD`, `Replay`). |
| **3. Attack Diagnosis** (Mechanistic Root Cause / Protocol Behavior) | Explain *how* the attack manifests (e.g., flood rate, ARP spoofing, unauthorized publish/connect, malformed payload) | **Partially Supported**: Packet-level timing (`frame.time_delta`), L2/L3 headers (`eth.*`, `ip.*`, `tcp.*`), and MQTT fields (`mqtt.msgtype`, `mqtt.topic`, `mqtt.msg`, `mqtt.qos`, `mqtt.dupflag`, `mqtt.retain`) expose *protocol mechanism* (e.g., 85.78% MQTT in `DoS` vs. 90.09% non-IP L2 in `mitm` vs. `CONNECT`/`DISCONNECT`/`PUBLISH` sequences in `intrusion`). | **Weak / Flow-Statistical Only**: No MQTT fields (`msgtype`, `topic`, `msg`, `clientid`, `dupflag`). Can only infer coarse flow patterns (e.g., `Slowite` low rate vs. `Flood` high packet rate), not MQTT protocol semantics. | **Supported** (known mutant mechanism + message token/timing/size/direction). |
| **4. Attack Localization** (Pinpointing Faulty Component / FSM State / Transition) | Identify *which* device, link, or **formal FSM state/transition** is compromised | **NOT Supported at FSM/Fault-Transition Level.** Only **L2/L3 host attribution** (MAC `eth.src`/`eth.dst` 100%, IP `ip.src`/`ip.dst` 95–100% except `mitm` 9.91%) is present; `mqtt.clientid` is sparse (`0%` in `mitm`, `0.69%` in `DoS`, `7.48%` in `intrusion`). **No FSM state/transition labels exist.** | **NOT Supported.** Contains no MAC, no `clientid`, no `topic`, and no FSM/component fault labels. `Normal` IPs are synthetic/byte-swapped (`83,283` unique `Src_IP`s). | **Supported** (exact mutant ID -> faulty TFSM transition/state). |

> **Critical Integrity Statement:** Neither `MQTT-UAD` nor `MQTTset` supports **formal fault localization** (identifying a violated TFSM state or transition). Localization in your thesis must rely on the **TFSM + mutant framework** (`models_fsm/`, `simulation/`), while `MQTT-UAD` provides **network/host-level attribution** (`eth.src`, `ip.src`, and sparse `mqtt.clientid`) and **protocol-level diagnosis** evidence.

---

## 2. Feature Mapping for Diagnosing Specific Attack Behaviors

### 2.1 Denial of Service (DoS / `Flood` / `MalariaDoS` / `Slowite`)
- **In MQTT-UAD (`DoS.csv`, 45,514 `DoS` rows)**:
  - **Timing & Volume**: `frame.time_delta`, `frame.time_delta_displayed` (inter-arrival burstiness), `frame.len`, `frame.cap_len`.
  - **MQTT Protocol Indicators**: `mqtt.msgtype` (37,626 `PUBLISH` = `3.0`; 361 `CONNACK` = `2.0`; 313 `CONNECT` = `1.0`; 339 `PUBREL` = `6.0`; 175 `DISCONNECT` = `14.0`), `mqtt.len`, `mqtt.qos`, `mqtt.dupflag`, `mqtt.retain`.
  - **Caveat**: `mqtt.topic` (`mqtt-malaria/...`, 5,902 unique) and `mqtt.clientid` (`beem.loadr-malaria-VirtualBox-...`, 313 unique) are generator artifacts and must **not** be used as raw string features if testing generalization.
- **In MQTTset (`Flood` 77,793; `MalariaDoS` 11,408; `Slowite` 3,044)**:
  - **Flow Rate & Duration**: `Flow_Duration`, `Flow_Pkts/s`, `Flow_Byts/s`, `Fwd_Pkts/s`, `Bwd_Pkts/s`, `Tot_Fwd_Pkts`, `Tot_Bwd_Pkts`.
  - **Inter-Arrival Time (IAT)**: `Flow_IAT_Mean`, `Flow_IAT_Std`, `Flow_IAT_Max`, `Flow_IAT_Min`, `Fwd_IAT_*`, `Bwd_IAT_*` (distinguishes high-rate `Flood` from low-rate `Slowite`).
  - **TCP Flags & Window**: `SYN_Flag_Cnt`, `FIN_Flag_Cnt`, `RST_Flag_Cnt`, `PSH_Flag_Cnt`, `ACK_Flag_Cnt`, `Init_Fwd_Win_Byts`.

### 2.2 Man-in-the-Middle (MITM)
- **In MQTT-UAD (`MitM.csv`, 3,855 `mitm` rows)**:
  - **Verified Diagnostic Signature**: **90.09% of `mitm` rows (3,473 / 3,855) have missing `ip.src`/`ip.dst`/`tcp.dstport`** while `eth.src`/`eth.dst` are 100% populated (3,505 from `b8:27:eb:73:32:30`), indicating **Layer-2 ARP spoofing** traffic rather than IP/TCP traffic.
  - **Remaining 9.91% (382 IP rows, 47 MQTT `PUBLISH` rows on `distance/ultrasonic1`)**: `frame.time_delta` (relay/interception delay), `ip.src`/`ip.dst` vs. `eth.src`/`eth.dst` consistency, `mqtt.msg`, `mqtt.qos`, `mqtt.dupflag`.
- **In MQTTset**: **Not represented** (`Cat` has no `MITM` class).

### 2.3 Replay-Like Behavior
- **In MQTT-UAD**:
  - **No explicit `Replay` label exists** in `type` (`normal`, `DoS`, `mitm`, `intrusion`).
  - However, **replay-relevant features** exist in MQTT-UAD: `mqtt.dupflag` (duplicate delivery flag), `mqtt.msgid` (packet identifier, though 100% NaN in `MitM.csv` and sparse in `DoS.csv`/`Intrusion.csv`), `mqtt.msg` + `mqtt.topic` + `frame.time_delta` (identical payload/topic repeated at anomalous intervals), and TCP/MAC sequence timing.
  - **Note on duplicates**: `DoS.csv` has 76,285 rows that are identical when ignoring `frame.number`/`frame.time_*` (reflecting repeated flood payloads), whereas `Intrusion.csv` has `PUBLISH` (`313`), `CONNECT` (`142`), `DISCONNECT` (`200`).
- **In MQTTset**: **Not represented** (no `Replay` label and no payload/sequence IDs).

### 2.4 Data Manipulation / Unauthorized Intrusion (`intrusion` / `Malformed`)
- **In MQTT-UAD (`Intrusion.csv`, 1,898 `intrusion` rows)**:
  - **Protocol & Payload Features**: `mqtt.msgtype` (45.10% populated: `PUBLISH` 313, `DISCONNECT` 200, `CONNACK` 198, `CONNECT` 142, `SUBSCRIBE` 2), `mqtt.topic` (`distance/ultrasonic1`, `light/rele1`, `motor`, `#` wildcard subscription), `mqtt.msg` (313 populated — can be inspected for out-of-range/injected sensor or actuator values), `mqtt.len`, `mqtt.retain`, `mqtt.qos`, `mqtt.clientid` (142 populated, e.g., `sensorPython`, `sensorUltrasonic`, `relay`).
- **In MQTTset (`Malformed`, 3,580 rows; `Bruteforce`, 4,547 rows)**:
  - Only **indirect flow-level side effects** are observable: `Pkt_Len_Min`, `Pkt_Len_Max`, `Pkt_Len_Mean`, `Pkt_Len_Std`, `Fwd_Pkt_Len_*`, `Bwd_Pkt_Len_*`, `RST_Flag_Cnt`, `FIN_Flag_Cnt`, `Down/Up_Ratio`. Actual malformed bytes or manipulated MQTT payloads are **absent**.

---

## 3. Device-Level Identity Assessment

- **MQTT-UAD**:
  - **MAC-level (`eth.src`, `eth.dst`)**: 100% populated across all 286,186 rows.
  - **IP-level (`ip.src`, `ip.dst`)**: 95–99.95% populated, except `mitm` (9.91% due to L2 ARP frames).
  - **MQTT `clientid` (`mqtt.clientid`)**: Only populated on `CONNECT` packets (0.00% in `mitm`, 0.02%–0.18% in `normal`, 0.69% in `DoS`, 7.48% in `intrusion`). Requires session-table propagation (`(ip.src, tcp.srcport, ip.dst, tcp.dstport) -> mqtt.clientid`) to associate subsequent `PUBLISH`/`SUBSCRIBE` packets with an MQTT client ID.
- **MQTTset**:
  - **Flow/IP-level only** (`Flow_ID`, `Src_IP`, `Dst_IP`, `Src_Port`, `Dst_Port`). No MAC, no `clientid`, and `Normal` IPs are synthetic/byte-swapped (`83,283` unique `Src_IP`s).

---

## 4. Recommended Next Implementation Stage (Awaiting Your Approval)

Before writing any model or preprocessing code, the logical **next stage** is **Phase 2: Leakage-Free Feature Engineering & Session-Aware Dataset Preparation**:

1. **MQTT-UAD Preprocessing Pipeline** (output to `datasets/processed/mqtt_uad/`):
   - Drop the 19–22 all-NaN columns, 4 constant columns, and capture-leakage columns (`frame.number`, `frame.time_epoch`, `frame.time_relative`, raw `mqtt.topic`/`mqtt.clientid` strings).
   - Build two feature views: (a) **Generalizable Behavioral View** (timing, packet sizes, TCP/MQTT flags, `mqtt.msgtype`, `mqtt.len`, IP/non-IP indicator) and (b) **Session-Attributed Metadata** (kept separate for post-hoc host/MAC attribution, not fed to the classifier).
   - Construct **chronological / contiguous-block train-val-test splits** per file so near-duplicate packets never leak across splits.
2. **MQTTset Preprocessing Pipeline** (output to `datasets/processed/mqttset/`, secondary benchmark):
   - Drop `Flow_ID`, `Src_IP`, `Dst_IP`, `Src_Port`, `Dst_Port`, `Protocol`, `Timestamp`, the 15 constant columns, and deduplicate on remaining features.
   - Replace/handle `±inf` in `Flow_Pkts/s` (328 rows) and `Flow_Byts/s` (193 rows).
3. **Align with Existing TFSM Pipeline (`models_fsm/`, `simulation/`, `ids/`)**:
   - Keep formal **fault localization** anchored in the TFSM + mutant two-head LSTM, and use MQTT-UAD to validate real-world **attack detection and protocol-level diagnosis** (`normal`, `DoS`, `mitm`, `intrusion`).

**Please confirm if you approve proceeding to Phase 2 (preprocessing & split generation) and which track(s) you want implemented first.**

