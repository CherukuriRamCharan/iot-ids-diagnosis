# Feature Selection and Leakage Report

> **Principle:** Not every available column is safe for model training. Columns that encode capture identity, lab-specific host addresses, synthetic generator strings, or collection-pipeline artifacts are excluded from predictive feature matrices (`X`) and preserved separately in `*_metadata.csv` for auditability and post-hoc alert attribution.

---

## 1. MQTT-UAD Feature Selection (`67` Raw Columns -> `33` Safe Features)

### 1.1 Retained Predictive Features (`33` Columns in `mqtt_uad_features.csv`)

| Group | Columns (`33` total) | Rationale |
|---|---|---|
| **Packet Timing & Size (4)** | `frame.time_delta`, `frame.time_delta_displayed`, `frame.cap_len`, `frame.len` | Captures inter-packet timing bursts (DoS) and frame-size anomalies without absolute capture epoch. |
| **Derived Protocol-Layer Indicators (6)** | `has_ip`, `has_tcp`, `is_mqtt_port`, `has_mqtt`, `has_mqtt_topic`, `has_mqtt_msg` | Captures protocol stack presence (e.g., Layer-2 ARP `mitm` frames have `has_ip=0, has_tcp=0`) without memorizing raw IP/MAC addresses. |
| **MQTT Numeric & Flag Fields (23)** | `mqtt.msgtype`, `mqtt.len`, `mqtt.qos`, `mqtt.dupflag`, `mqtt.retain`, `mqtt.kalive`, `mqtt.ver`, `mqtt.proto_len`, `mqtt.clientid_len`, `mqtt.topic_len`, `mqtt.msgid`, `mqtt.sub.qos`, `mqtt.suback.qos`, `mqtt.conack.val`, `mqtt.conack.flags.reserved`, `mqtt.conack.flags.sp`, `mqtt.conflag.cleansess`, `mqtt.conflag.passwd`, `mqtt.conflag.qos`, `mqtt.conflag.reserved`, `mqtt.conflag.retain`, `mqtt.conflag.uname`, `mqtt.conflag.willflag` | Captures MQTT protocol state transitions and flags (`CONNECT`, `CONNACK`, `PUBLISH`, `DISCONNECT`, `dupflag`, `retain`, `qos`). Missing values encoded as `-1.0` alongside `has_mqtt`. |

### 1.2 Excluded Columns (`39` Raw Columns) and Reasons

| Exclusion Category | Count | Columns | Exact Reason |
|---|---:|---|---|
| **100% Missing (All-NaN)** | 19 | `frame.time_invalid`, `frame.coloring_rule.name`, `frame.coloring_rule.string`, `frame.comment`, `frame.comment.expert`, `frame.file_off`, `frame.incomplete`, `frame.interface_id`, `frame.interface_name`, `frame.link_nr`, `frame.md5_hash`, `mqtt.passwd`, `mqtt.passwd_len`, `mqtt.username`, `mqtt.username_len`, `mqtt.willmsg`, `mqtt.willmsg_len`, `mqtt.willtopic`, `mqtt.willtopic_len` | Zero non-null values across all 286,186 rows. |
| **Constant (Zero Variance)** | 4 | `frame.encap_type` (`1`), `frame.ignored` (`0`), `frame.marked` (`0`), `frame.offset_shift` (`0.0`) | Single unique value across all 286,186 rows. |
| **Capture & Time Leakage** | 3 | `frame.number`, `frame.time_epoch`, `frame.time_relative` | Uniquely identify which capture file (`DoS.csv` Mar 2018 vs. `MitM.csv` Jun 2018 vs. `Intrusion.csv` Sep 2018) and attack window a row came from. |
| **Host/Lab Identity Leakage** | 6 | `ip.src`, `ip.dst`, `eth.src`, `eth.dst`, `tcp.srcport`, `tcp.dstport` | Single-lab host memorization (e.g., `b8:27:eb:73:32:30` accounts for 3,505/3,855 `mitm` frames; `192.168.1.196 -> 192.168.1.171` dominates `DoS`). Retained in `mqtt_uad_metadata.csv` for post-hoc attribution only. |
| **Generator / String Leakage** | 7 | `mqtt.topic`, `mqtt.clientid`, `mqtt.msg`, `mqtt.protoname`, `mqtt.hdrflags`, `mqtt.conflags`, `mqtt.conack.flags` | `mqtt.topic` (`mqtt-malaria/...`) and `mqtt.clientid` (`beem.loadr-malaria-VirtualBox-...`) appear exclusively in `DoS`; raw strings would cause trivial memorization. Retained in `mqtt_uad_metadata.csv` as application-layer evidence. |

---

## 2. MQTTset Feature Selection (`85` Raw Columns -> `62` Benchmark Features)

### 2.1 Excluded Shortcut-Leakage & Constant Columns (`22` Columns)

| Exclusion Category | Count | Columns | Verified Leakage / Quality Evidence |
|---|---:|---|---|
| **Shortcut & Identifier Leakage** | 7 | `Flow_ID`, `Src_IP`, `Dst_IP`, `Src_Port`, `Dst_Port`, `Protocol`, `Timestamp` | **100% of `Normal` rows** (440,699) have `Protocol=0, Dst_Port=0` and 83,283 byte-swapped `Src_IP`s, whereas **100% of `Anomaly` rows** (100,372) have `Protocol=6, Dst_Port=1883`. Furthermore, every `Cat` has disjoint timestamps and/or IP pairs (`Flood`: `192.168.1.73 -> 192.168.1.105` on `15/07/2021`; `Slowite`: `10.16.120.44 -> 10.16.120.72` on `24/07/2019`; `MalariaDoS`/`Bruteforce`/`Malformed`: `192.168.1.90 <-> .91` on separate dates). |
| **Constant (Zero Variance)** | 15 | `Bwd_PSH_Flags`, `Fwd_URG_Flags`, `Bwd_URG_Flags`, `URG_Flag_Cnt`, `CWE_Flag_Count`, `ECE_Flag_Cnt`, `Fwd_Byts/b_Avg`, `Fwd_Pkts/b_Avg`, `Fwd_Blk_Rate_Avg`, `Bwd_Byts/b_Avg`, `Subflow_Bwd_Pkts`, `Active_Mean`, `Active_Std`, `Active_Max`, `Active_Min` | Single value `0` across all 541,071 rows. |

### 2.2 Retained Predictive Features (`62` Columns in `mqttset_features.csv`)
- **61 statistical flow features**: `Flow_Duration`, `Tot_Fwd_Pkts`, `Tot_Bwd_Pkts`, `TotLen_Fwd_Pkts`, `TotLen_Bwd_Pkts`, `Fwd_Pkt_Len_*` (4), `Bwd_Pkt_Len_*` (4), `Flow_Byts/s`, `Flow_Pkts/s`, `Flow_IAT_*` (4), `Fwd_IAT_*` (5), `Bwd_IAT_*` (5), `Fwd_PSH_Flags`, `Fwd_Header_Len`, `Bwd_Header_Len`, `Fwd_Pkts/s`, `Bwd_Pkts/s`, `Pkt_Len_*` (5), `FIN_Flag_Cnt`, `SYN_Flag_Cnt`, `RST_Flag_Cnt`, `PSH_Flag_Cnt`, `ACK_Flag_Cnt`, `Down/Up_Ratio`, `Pkt_Size_Avg`, `Fwd_Seg_Size_Avg`, `Bwd_Seg_Size_Avg`, `Bwd_Pkts/b_Avg`, `Bwd_Blk_Rate_Avg`, `Subflow_Fwd_Pkts`, `Subflow_Fwd_Byts`, `Subflow_Bwd_Byts`, `Init_Fwd_Win_Byts`, `Init_Bwd_Win_Byts`, `Fwd_Act_Data_Pkts`, `Fwd_Seg_Size_Min`, `Idle_*` (4).
- **1 derived infinity indicator**: `has_inf_rate` (`1` for rows where `Flow_Pkts/s` or `Flow_Byts/s` was `±inf` before train-median imputation).

### 2.3 Remaining Residual Risk in MQTTset
- Even after removing `Protocol`, `Dst_Port`, `Src_IP`, `Dst_IP`, and `Timestamp`, `Subflow_Bwd_Pkts` is constant `0` and `Normal` flows in the raw capture often have `Tot_Bwd_Pkts = 0` and `TotLen_Fwd_Pkts = 0`. Therefore, **MQTTset must be treated strictly as a secondary, caveated flow-level benchmark**, never as proof of real-world MQTT generalization.

