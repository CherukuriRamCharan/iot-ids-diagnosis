# Phase 2 Preprocessing Report

> **Project:** Attack Diagnosis and Localization in IoT Intrusion Detection Using Formal Modeling and Deep Learning  
> **Validation Status:** `OVERALL PHASE 2 VALIDATION PASSED: True` (verified via `features/validate_phase2.py`, `SEED = 42`)

---

## 1. Overview of Generated Outputs

All raw files in `datasets/raw/` remain untouched. Reproducible scripts were added under `features/`, and processed artifacts were written to `datasets/processed/`:

| Track | Script | Output Files |
|---|---|---|
| **Primary: MQTT-UAD** | `features/preprocess_mqtt_uad.py` | `datasets/processed/mqtt_uad/mqtt_uad_features.csv`, `mqtt_uad_metadata.csv`, `mqtt_uad_manifest.json` |
| **Secondary: MQTTset** | `features/preprocess_mqttset.py` | `datasets/processed/mqttset/mqttset_features.csv`, `mqttset_metadata.csv`, `mqttset_manifest.json` |
| **Validation** | `features/validate_phase2.py` | `datasets/processed/phase2_validation_summary.txt` |

---

## 2. Track 1: MQTT-UAD Preprocessing (`datasets/processed/mqtt_uad/`)

### 2.1 Row Counts & Label Preservation
- **Input**: `DoS.csv` (94,625), `MitM.csv` (110,668), `Intrusion.csv` (80,893) -> **286,186 total rows**.
- **Output**: **286,186 rows** (no rows dropped; near-duplicates are handled via chronological contiguous block splitting rather than deleting packet-stream rows, preserving packet-sequence timing).
- **Label distribution (`type`) — unchanged**:
  - `normal`: 234,919 (82.09%)
  - `DoS`: 45,514 (15.90%)
  - `mitm`: 3,855 (1.35%)
  - `intrusion`: 1,898 (0.66%)

### 2.2 Exact Transformations Applied
1. **Source Traceability**: Added `source_file` and 0-based `source_row_idx` to both `mqtt_uad_features.csv` and `mqtt_uad_metadata.csv`.
2. **Column Separation**:
   - **Dropped (23 columns)**: 19 all-NaN columns across all 3 files + 4 constant columns (`frame.encap_type=1`, `frame.ignored=0`, `frame.marked=0`, `frame.offset_shift=0.0`).
   - **Moved to `mqtt_uad_metadata.csv` (16 columns)**: 9 identity/time/port columns (`frame.number`, `frame.time_epoch`, `frame.time_relative`, `ip.src`, `ip.dst`, `eth.src`, `eth.dst`, `tcp.srcport`, `tcp.dstport`) and 7 MQTT string/hex evidence columns (`mqtt.topic`, `mqtt.clientid`, `mqtt.msg`, `mqtt.protoname`, `mqtt.hdrflags`, `mqtt.conflags`, `mqtt.conack.flags`).
3. **Derived Protocol-Layer Indicators (6 binary columns)**:
   - `has_ip` (`ip.src` non-null), `has_tcp` (`tcp.srcport` non-null), `is_mqtt_port` (`tcp.srcport == 1883` or `tcp.dstport == 1883`), `has_mqtt` (`mqtt.msgtype` non-null), `has_mqtt_topic` (`mqtt.topic` non-null), `has_mqtt_msg` (`mqtt.msg` non-null).
4. **Missing & Invalid Value Handling (No Fabricated MQTT Data)**:
   - Coerced `±inf` to `NaN` in all 27 numeric packet/MQTT columns.
   - Filled missing numeric MQTT/packet fields with sentinel `-1.0` (outside valid MQTT protocol ranges `0..15` / `>=0`) paired with explicit `has_mqtt` / `has_mqtt_topic` / `has_mqtt_msg` flags so absence of MQTT data is never confused with `0`.
5. **Validation**: `33` predictive feature columns; `0` NaNs; `0` `±inf` values.

### 2.3 Application-Layer Evidence Sparsity by Attack Category
- **`DoS`**: Strong MQTT presence (`has_mqtt` 85.78%, `mqtt.topic` 82.38%, `mqtt.msg` 82.38%), but `mqtt.topic` (`mqtt-malaria/...`) and `mqtt.clientid` (`beem.loadr-malaria-VirtualBox-...`, 0.69%) are synthetic load-generator strings.
- **`mitm`**: **Lacks application-layer evidence** (`mqtt.msgtype`/`topic`/`msg` populated in only **47 / 3,855 = 1.22%** of rows; `mqtt.clientid` **0.00%**; `ip.src`/`ip.dst` **9.91%**). ~90.09% of `mitm` frames are Layer-2 ARP/non-IP packets.
- **`intrusion`**: Moderate MQTT presence (`mqtt.msgtype` **45.10%**, `mqtt.topic` **16.60%**, `mqtt.msg` **16.49%**, `mqtt.clientid` **7.48%**).
- **`normal`**: Mostly non-MQTT background traffic (`mqtt.topic`/`msg` 0.48%–2.19%).

---

## 3. Track 2: MQTTset Preprocessing (`datasets/processed/mqttset/`)

### 3.1 Row Counts & Deduplication Impact
- **Input**: `MQTTset.csv` (**541,071 rows**, 85 columns, streamed in `50,000`-row chunks).
- **Output**: **114,517 rows** (**426,554 duplicate behavioral feature rows dropped** after excluding `Flow_ID`, `Timestamp`, `Src_IP`, `Dst_IP`, `Src_Port`, `Dst_Port`, `Protocol`).
- **Original `source_row_idx` mapping**: Preserved in `mqttset_metadata.csv` alongside `Flow_ID`, `Src_IP`, `Dst_IP`, `Src_Port`, `Dst_Port`, `Protocol`, `Timestamp`, `label`, `Cat`, and `split`.

| `Cat` | Raw Rows | Processed Rows (Deduplicated) | Duplicates Dropped |
|---|---:|---:|---:|
| `Normal` | 440,699 (81.45%) | 20,397 (17.81%) | 420,302 (95.37% of raw `Normal`) |
| `Flood` | 77,793 (14.38%) | 76,904 (67.16%) | 889 |
| `MalariaDoS` | 11,408 (2.11%) | 10,852 (9.48%) | 556 |
| `Bruteforce` | 4,547 (0.84%) | 3,066 (2.68%) | 1,481 |
| `Malformed` | 3,580 (0.66%) | 2,510 (2.19%) | 1,070 |
| `Slowite` | 3,044 (0.56%) | 788 (0.69%) | 2,256 |
| **Total** | **541,071** | **114,517** | **426,554** |

> **Key Finding:** 95.37% of raw `Normal` rows in `MQTTset.csv` were identical behavioral flow vectors differing only in randomized/anonymized `Src_IP`, `Flow_ID`, and `Timestamp`. Deduplication prevents evaluating models on 420k+ repeated `Normal` vectors.

### 3.2 Exact Transformations Applied
1. **Excluded Leakage/Identifier Columns (7)**: `Flow_ID`, `Src_IP`, `Dst_IP`, `Src_Port`, `Dst_Port`, `Protocol`, `Timestamp` (stored in `mqttset_metadata.csv`).
2. **Dropped Constant Columns (15)**: `Bwd_PSH_Flags`, `Fwd_URG_Flags`, `Bwd_URG_Flags`, `URG_Flag_Cnt`, `CWE_Flag_Count`, `ECE_Flag_Cnt`, `Fwd_Byts/b_Avg`, `Fwd_Pkts/b_Avg`, `Fwd_Blk_Rate_Avg`, `Bwd_Byts/b_Avg`, `Subflow_Bwd_Pkts`, `Active_Mean`, `Active_Std`, `Active_Max`, `Active_Min`.
3. **Infinity Handling (Zero Train->Test Leakage)**:
   - Flagged `328` `±inf` rows in `Flow_Pkts/s` and `193` `±inf` rows in `Flow_Byts/s` with binary indicator `has_inf_rate`.
   - Replaced `±inf` with `NaN` and imputed using **train-split medians only** (`Flow_Byts/s`: `54,074,074.07`, `Flow_Pkts/s`: `52,631.58`).
4. **Validation**: `62` predictive feature columns (`61` flow features + `has_inf_rate`); `0` NaNs; `0` `±inf` values.
5. **Scope Caveat**: `MQTTset` contains **zero** MQTT topic, payload, client-ID, or formal fault-localization fields.

---

## 4. Reproducibility Commands & Verified Validation Summary

### 4.1 Exact Commands to Reproduce
Run from `C:\Users\cheru\Downloads\iot-ids-diagnosis\iot-ids-diagnosis` (fixed `SEED = 42`):

```cmd
python features\preprocess_mqtt_uad.py && python features\preprocess_mqttset.py && python features\validate_phase2.py
```

### 4.2 Verified Validation Results (`datasets/processed/phase2_validation_summary.txt`)
- **MQTT-UAD**: `286,186` processed rows (`normal`: `234,919`, `DoS`: `45,514`, `mitm`: `3,855`, `intrusion`: `1,898`), `33` feature columns, `0` NaNs, `0` `±inf` values, contiguous non-overlapping chronological splits (`train`: `200,329`, `val`: `42,926`, `test`: `42,931`), `MQTT-UAD VALIDATION PASSED: True`.
- **MQTTset**: `541,071` raw -> `114,517` processed rows (`426,554` duplicate feature rows dropped), `62` feature columns, `0` NaNs, `0` `±inf` values, `Flow_ID` overlap across `train`/`val`/`test` = `0`, `MQTTSET VALIDATION PASSED: True`.
- **Overall**: `OVERALL PHASE 2 VALIDATION PASSED: True`. (Note: This validates data integrity, schema, and split hygiene only; it does not evaluate model accuracy or attack-localization accuracy.)

