# Stage 3B.5 — Independent Baseline Results and Feature Leakage Audit

> **Audit Scope:** Independent read-only audit of Stage 3B results (`results/stage_3b_baselines/`), Stage 3A feature selection (`ids/datasets_real.py`), Phase 2 manifests (`datasets/processed/*/*_manifest.json`), and diagnostic single-feature separability & feature-importance checks.  
> **Overall Verdict:**
> - **MQTT-UAD results:** **Trustworthy (within-capture chronological holdout)** — realistic minority-class difficulty (`mitm`, `intrusion`), no target/identifier leakage, and clear protocol-level behavior.
> - **MQTTset results:** **Unsuitable for strong generalization or real-world IDS claims (Conditionally trustworthy only as a synthetic/artifact-prone secondary flow benchmark)** — `1.0000` binary macro-F1 is driven by systemic TCP-header and flow-duration capture-pipeline artifacts between `Normal` and `Anomaly`.

---

## 1. Exact Predictive Feature Lists Verified

### 1.1 MQTTset (`62` Features)
Verified directly from `load_mqttset()` and `mqttset_manifest.json`:
```text
Flow_Duration, Tot_Fwd_Pkts, Tot_Bwd_Pkts, TotLen_Fwd_Pkts, TotLen_Bwd_Pkts,
Fwd_Pkt_Len_Max, Fwd_Pkt_Len_Min, Fwd_Pkt_Len_Mean, Fwd_Pkt_Len_Std,
Bwd_Pkt_Len_Max, Bwd_Pkt_Len_Min, Bwd_Pkt_Len_Mean, Bwd_Pkt_Len_Std,
Flow_Byts/s, Flow_Pkts/s, Flow_IAT_Mean, Flow_IAT_Std, Flow_IAT_Max, Flow_IAT_Min,
Fwd_IAT_Tot, Fwd_IAT_Mean, Fwd_IAT_Std, Fwd_IAT_Max, Fwd_IAT_Min,
Bwd_IAT_Tot, Bwd_IAT_Mean, Bwd_IAT_Std, Bwd_IAT_Max, Bwd_IAT_Min,
Fwd_PSH_Flags, Fwd_Header_Len, Bwd_Header_Len, Fwd_Pkts/s, Bwd_Pkts/s,
Pkt_Len_Min, Pkt_Len_Max, Pkt_Len_Mean, Pkt_Len_Std, Pkt_Len_Var,
FIN_Flag_Cnt, SYN_Flag_Cnt, RST_Flag_Cnt, PSH_Flag_Cnt, ACK_Flag_Cnt,
Down/Up_Ratio, Pkt_Size_Avg, Fwd_Seg_Size_Avg, Bwd_Seg_Size_Avg,
Bwd_Pkts/b_Avg, Bwd_Blk_Rate_Avg, Subflow_Fwd_Pkts, Subflow_Fwd_Byts,
Subflow_Bwd_Byts, Init_Fwd_Win_Byts, Init_Bwd_Win_Byts, Fwd_Act_Data_Pkts,
Fwd_Seg_Size_Min, Idle_Mean, Idle_Std, Idle_Max, Idle_Min, has_inf_rate
```
- **Target & Identifier Leakage Check:** Confirmed **absent** from `X`: `Protocol`, `Dst_Port`, `Src_Port`, `Src_IP`, `Dst_IP`, `Flow_ID`, `Timestamp`, `label`, `Cat`, `source_row_idx`, `split`.
- **Binary Label Creation Check:** `y_all = (df["label"].astype(str) != "Normal").astype(np.int64)` uses `label` strictly for `y`, never entering `X`.

### 1.2 MQTT-UAD (`33` Features)
Verified directly from `load_mqtt_uad()` and `mqtt_uad_manifest.json`:
- `frame.time_delta`, `frame.time_delta_displayed`, `frame.cap_len`, `frame.len`, `has_ip`, `has_tcp`, `is_mqtt_port`, `has_mqtt`, `has_mqtt_topic`, `has_mqtt_msg`, plus `23` numeric/flag `mqtt.*` columns.
- Confirmed **absent** from `X`: `frame.number`, `frame.time_epoch`, `frame.time_relative`, `ip.*`, `eth.*`, `tcp.*port`, `mqtt.topic`, `mqtt.clientid`, `mqtt.msg`, `source_file`, `source_row_idx`, `type`, `split`.

---

## 2. Why MQTTset Achieved `1.0000` Binary Macro-F1 (Suspicious Proxy Analysis)

Even though `Protocol` and `Dst_Port` were excluded, **two TCP-header features act as 100% exact proxies for `Protocol=0` (`Normal`) vs. `Protocol=6` (`Anomaly`)**, and **flow-duration / IAT / packet-rate features** separate the two capture pipelines at `0.9996` macro-F1 on their own.

### 2.1 Single-Feature Decision-Stump (Depth = 1) Separability on MQTTset Test Split

| Feature | Stump Macro-F1 | Stump Accuracy | Verified Mechanism |
|---|---:|---:|---|
| `Fwd_Seg_Size_Min` | **1.0000** | **1.0000** | **100.00% of `Normal` = `0.0`** (min=0, max=0); **0.00% of `Anomaly` = `0.0`** (min=20, max=44) across train, val, and test. |
| `Fwd_Header_Len` | **1.0000** | **1.0000** | **100.00% of `Normal` = `0.0`** (min=0, max=0); **0.00% of `Anomaly` = `0.0`** (min=20, max=1760) across train, val, and test. |
| `Fwd_IAT_Tot` | **0.9996** | **0.9998** | `Normal` test median = `5,497,968 µs` (max `120 s`); `Anomaly` test median = `31 µs` (max `1,200 µs`). |
| `Flow_Pkts/s` | **0.9996** | **0.9998** | `Normal` test median = `0.400 pkt/s`; `Anomaly` test median = `59,322 pkt/s` (`min = 1,666.7`). |
| `Flow_Duration` / `Flow_IAT_Mean` / `Flow_IAT_Max` | **0.9996** | **0.9998** | Same multi-second vs. sub-millisecond capture-pipeline separation. |
| `ACK_Flag_Cnt` | **0.9986** | **0.9992** | `Normal` (`Protocol=0`) has `0` TCP ACK flags; `Anomaly` (`Protocol=6`) has TCP flags. |
| `Init_Fwd_Win_Byts` | **0.9985** | **0.9992** | `Normal` (`Protocol=0`) has `0` TCP window bytes; `Anomaly` (`Protocol=6`) has TCP window bytes. |

### 2.2 Feature Importance Findings (MQTTset)
- **Binary DecisionTree**: **`Fwd_Header_Len` alone has importance `1.0000`** (all other `61` features = `0.0000`), confirming the tree makes a single split (`Fwd_Header_Len <= 10`) to achieve `1.0000` test accuracy.
- **Binary RandomForest**: Top importances are `Flow_Pkts/s` (`0.1211`), `ACK_Flag_Cnt` (`0.1099`), `Fwd_Seg_Size_Min` (`0.1010`), `Fwd_Header_Len` (`0.0912`), `Init_Fwd_Win_Byts` (`0.0708`), `Flow_IAT_Max` (`0.0699`).
- **6-Class RandomForest / DecisionTree**: Driven by `Fwd_Seg_Size_Avg`, `Fwd_Seg_Size_Min`, `Idle_Max`, `Init_Fwd_Win_Byts`, `Pkt_Len_Max`, `Fwd_Header_Len`, `Fwd_Pkt_Len_Mean`.

### 2.3 Diagnostic Ablation Assessment
- Removing only `Fwd_Header_Len` and `Fwd_Seg_Size_Min` would **not** eliminate the artifact, because `ACK_Flag_Cnt` (`0.9986`), `Init_Fwd_Win_Byts` (`0.9985`), `Fwd_IAT_Tot` (`0.9996`), and `Flow_Pkts/s` (`0.9996`) independently separate `Normal` from `Anomaly` at `99.85%–99.98%`.
- **Conclusion:** The `1.0000` binary score on `MQTTset` is **not** a bug in Stage 3A/3B code; it is an **inherent artifact of the raw `MQTTset.csv` dataset**, where `Normal` flows (`Protocol=0`, zero TCP headers, multi-second durations) and `Anomaly` flows (`Protocol=6`, TCP port 1883, sub-millisecond bursts) were generated/exported differently.

---

## 3. MQTT-UAD Independent Review

### 3.1 Minority-Class & MQTT-Only Behavior
- **Full Stream 4-Class**:
  - `RandomForest` Macro-F1 = `0.6098 ± 0.0009` (Accuracy = `0.9696`); `DecisionTree` Macro-F1 = `0.6088 ± 0.0000` (Accuracy = `0.9746`).
  - `normal` (`0.9881` RF F1) and `DoS` (`0.9509` RF F1) perform strongly, whereas minority classes **`mitm` (`162` test rows, F1 = `0.2597` RF / `0.2910` DT)** and **`intrusion` (`112` test rows, F1 = `0.2405` RF / `0.2012` DT)** suffer from low precision (`0.1653`, `0.1411`) due to false positives against `normal` (`36,170` rows).
- **MQTT-Only 4-Class (`has_mqtt == 1`)**:
  - Filtering to MQTT packets lifts `intrusion` F1 from `0.2405` to **`0.6738` (RF)** / `0.6310` (DT) and `DoS` F1 to **`0.9991`**.
  - **Critical `mitm` limitation**: `mitm` has only **`13` test samples** in `MQTT-Only` (`90.09%` of `mitm` is Layer-2 ARP/non-IP), yielding **F1 = `0.1333`** (`recall = 0.0769`, 1/13 detected). Thus `MQTT-Only` cannot reliably evaluate `mitm`.

### 3.2 Shortcut Check on MQTT-UAD
- **Full-Stream Top RF Importances**: `frame.cap_len` (`0.1656`), `is_mqtt_port` (`0.1517`), `frame.len` (`0.1480`), `frame.time_delta_displayed` (`0.0711`), `frame.time_delta` (`0.0699`), `has_ip` (`0.0693`), `mqtt.topic_len` (`0.0629`), `mqtt.len` (`0.0583`).
- **MQTT-Only Top RF Importances**: `frame.time_delta` (`0.2389`), `frame.time_delta_displayed` (`0.1927`), `mqtt.topic_len` (`0.1163`), `frame.len` (`0.1124`), `frame.cap_len` (`0.0994`), `mqtt.len` (`0.0952`), `mqtt.msgtype` (`0.0411`).
- **All-NaN `MitM.csv` columns check**: `mqtt.msgid` (`0.0012` DT) and `mqtt.sub.qos` (`0.0006` DT) have negligible importance (`< 0.002`), confirming models are **not** exploiting `MitM.csv` missingness as a primary shortcut.
- **Reminder**: All MQTT-UAD results remain **within-capture chronological holdout** performance, not unseen-capture generalization.

---

## 4. Trustworthiness & Recommended Corrections Before Stage 3C (LSTM)

1. **MQTT-UAD Results**: **Trustworthy as within-capture chronological baselines.** Keep `MQTT-UAD` as the **primary** dataset for Stage 3C (LSTM) and Stage 3D (Transformer), where temporal packet sequences are expected to help `mitm` and `intrusion` precision.
2. **MQTTset Results**: **Not trustworthy for strong real-world binary detection claims.** Keep `MQTTset` strictly as a **secondary, caveated 6-class flow benchmark** in the thesis, explicitly disclosing that `Normal` (`Protocol=0`) vs. `Anomaly` (`Protocol=6`) is trivially separated (`1.0000` F1) by TCP-header presence (`Fwd_Header_Len`, `Fwd_Seg_Size_Min`) and capture-pipeline duration/rate differences (`Fwd_IAT_Tot`, `Flow_Pkts/s`).
3. **Corrections for Stage 3C/3D**:
   - Report **both** Full-Stream (primary for `mitm` L2 detection) and MQTT-Only (`has_mqtt == 1`, noting `n=13` `mitm` test support) on MQTT-UAD.
   - Do **not** present `MQTTset` `1.0000` binary accuracy as evidence of IDS superiority; frame it honestly as a dataset-artifact finding.

