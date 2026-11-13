# Deep Dataset Analysis: MQTT-UAD and MQTTset

> **Scope & Grounding:** All counts, percentages, column names, and value distributions in this document are **verified findings** from read-only chunked (`50,000` rows/chunk) inspection of the raw CSVs in `datasets/raw/mqtt_uad/` and `datasets/raw/mqttset/` (recorded in `docs/MQTT_UAD_DATASET_AUDIT.md`, `docs/MQTTSET_DATASET_AUDIT.md`, and the deep audit run). Any interpretation beyond raw counts is explicitly labeled as an **Assumption / Inference**.

---

## 1. Executive Summary

| Property | MQTT-UAD (`DoS.csv`, `MitM.csv`, `Intrusion.csv`) | MQTTset (`MQTTset.csv`) |
|---|---|---|
| **Granularity** | Packet-level (Wireshark/TShark 67 columns) | Flow-level (CICFlowMeter-style 85 columns) |
| **Total Rows** | 286,186 | 541,071 |
| **Label Column(s)** | `type` (`normal`, `DoS`, `mitm`, `intrusion`) | `label` (`Normal`, `Anomaly`), `Cat` (`Normal`, `Flood`, `MalariaDoS`, `Bruteforce`, `Malformed`, `Slowite`) |
| **Attack Ratio** | 17.91% attack (51,267 / 286,186) | 18.55% anomaly (100,372 / 541,071) |
| **MQTT App-Layer Fields** | Present (`mqtt.topic`, `mqtt.msg`, `mqtt.msgtype`, `mqtt.clientid`, flags), sparse on `normal` & `mitm` | **Absent** (0 MQTT-specific columns) |
| **Device Identity** | IP + MAC (`eth.src`/`eth.dst`) + partial `mqtt.clientid` | IP + Port + `Flow_ID` only (no MAC, no `clientid`) |
| **Critical Quality / Leakage Issues** | 19–22 all-NaN cols; 4 constant cols; 32,365 exact duplicates; synthetic `mqtt-malaria` topics in `DoS`; 90.1% missing IPs in `mitm` | **Severe shortcut leakage**: `Normal` has `Protocol=0, Dst_Port=0`; disjoint timestamps & IP pairs per `Cat`; 15 constant cols; `±inf` values |

---

## 2. MQTT-UAD Deep Inspection

### 2.1 Label Categories and Exact Distributions (`type`)

**Verified Findings:**

| File | `normal` | Attack Class | Attack Rows | Attack % | Total Rows | Epoch Range (`frame.time_epoch`) |
|---|---:|---|---:|---:|---:|---|
| `DoS.csv` | 49,111 (51.90%) | `DoS` | 45,514 | 48.10% | 94,625 | `1522235370.391` – `1522236312.444` (942.1 s) |
| `MitM.csv` | 106,813 (96.52%) | `mitm` | 3,855 | 3.48% | 110,668 | `1529667811.198` – `1529672517.769` (4,706.6 s) |
| `Intrusion.csv` | 78,995 (97.65%) | `intrusion` | 1,898 | 2.35% | 80,893 | `1535971845.729` – `1535977010.137` (5,164.4 s) |
| **Combined** | **234,919 (82.09%)** | **All Attacks** | **51,267** | **17.91%** | **286,186** | Three disjoint capture dates (Mar, Jun, Sep 2018) |

Combined `type` breakdown:
- `normal`: 234,919 (82.09%)
- `DoS`: 45,514 (15.90%)
- `mitm`: 3,855 (1.35%)
- `intrusion`: 1,898 (0.66%)

**Usability of Attack-Specific Labels:**
- **Yes**, MQTT-UAD provides usable attack-specific labels (`DoS`, `mitm`, `intrusion`, `normal`) via the `type` column.
- However, each attack class originates from a **separate capture file** taken months apart (March 2018, June 2018, September 2018), creating severe class imbalance for `mitm` (1.35%) and `intrusion` (0.66%) when combined.

### 2.2 Field Population by Class (`normal` vs. Attack)

**Verified Findings (populated counts / total class rows):**

| Column | `DoS.csv` (`DoS`: 45,514) | `DoS.csv` (`normal`: 49,111) | `MitM.csv` (`mitm`: 3,855) | `MitM.csv` (`normal`: 106,813) | `Intrusion.csv` (`intrusion`: 1,898) | `Intrusion.csv` (`normal`: 78,995) |
|---|---:|---:|---:|---:|---:|---:|
| `frame.time_epoch` | 45,514 (100.00%) | 49,111 (100.00%) | 3,855 (100.00%) | 106,813 (100.00%) | 1,898 (100.00%) | 78,995 (100.00%) |
| `eth.src` / `eth.dst` | 45,514 (100.00%) | 49,111 (100.00%) | 3,855 (100.00%) | 106,813 (100.00%) | 1,898 (100.00%) | 78,995 (100.00%) |
| `ip.src` / `ip.dst` | 45,491 (99.95%) | 48,101 (97.94%) | **382 (9.91%)** | 102,053 (95.54%) | 1,896 (99.89%) | 75,210 (95.21%) |
| `tcp.srcport` / `tcp.dstport` | 45,460 (99.88%) | 43,000 (87.56%) | **380 (9.86%)** | 99,134 (92.81%) | 1,894 (99.79%) | 69,863 (88.44%) |
| `mqtt.msgtype` | 39,044 (85.78%) | 491 (1.00%) | 47 (1.22%) | 3,572 (3.34%) | 856 (45.10%) | 3,962 (5.02%) |
| `mqtt.topic` | 37,496 (82.38%) | 241 (0.49%) | 47 (1.22%) | 1,988 (1.86%) | 315 (16.60%) | 1,732 (2.19%) |
| `mqtt.msg` | 37,496 (82.38%) | 237 (0.48%) | 47 (1.22%) | 1,988 (1.86%) | 313 (16.49%) | 1,708 (2.16%) |
| `mqtt.qos` / `dupflag` / `retain` | 38,227 (83.99%) | 237 (0.48%) | 47 (1.22%) | 1,988 (1.86%) | 313 (16.49%) | 1,708–1,714 (2.16–2.17%) |
| `mqtt.clientid` | 313 (0.69%) | 10 (0.02%) | **0 (0.00%)** | 4 (0.00%) | 142 (7.48%) | 139 (0.18%) |
| `mqtt.username` / `mqtt.passwd` | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |

**Key Observations:**
1. **`normal` traffic is mostly non-MQTT background traffic**: Only 0.48%–2.19% of `normal` packets contain `mqtt.topic` / `mqtt.msg`; the majority are HTTPS (`tcp.dstport=443.0`), SSH (`22`), HTTP (`80`), and other local/internet traffic.
2. **`mitm` is ~90% Layer-2 (non-IP) traffic**: 3,473 of 3,855 `mitm` rows lack `ip.src`/`ip.dst`, while `eth.src` is 100% populated (3,505 from `b8:27:eb:73:32:30`). **Inference:** Those ~90.1% non-IP frames are ARP spoofing packets used to establish the MitM position, while only 47 `mitm` packets are actual MQTT `PUBLISH` (`mqtt.msgtype=3.0`) on `distance/ultrasonic1`.
3. **`DoS` uses synthetic `mqtt-malaria` flood traffic**: 5,902 unique `mqtt.topic` values matching `mqtt-malaria/beem.loadr-malaria-VirtualBox-.../data/...` and 313 unique `mqtt.clientid` values matching `beem.loadr-malaria-VirtualBox-...`, sent primarily from `192.168.1.196` (`40,751` packets) to `192.168.1.171:1883` (`40,767` packets).
4. **`intrusion` contains real MQTT protocol interactions**: 856/1,898 packets (45.10%) have `mqtt.msgtype` (`3.0`: 313, `14.0` DISCONNECT: 200, `2.0` CONNACK: 198, `1.0` CONNECT: 142, `8.0` SUBSCRIBE: 2, `13.0`: 1), with topics `distance/ultrasonic1` (142), `light/rele1` (134), `motor` (37), `#` (2), and client IDs such as `sensorUltrasonic`, `sensorPython`, `sensorLuz`, `relay`, `sensorMotor`.

### 2.3 Device Identity vs. IP/MAC-Level Identification

**Verified Findings:**
- `eth.src` and `eth.dst` (MAC addresses) are 100% populated across all 286,186 rows (8–10 unique MACs per class).
- `ip.src` and `ip.dst` are 95–99.95% populated except in `mitm` (9.91%).
- `mqtt.clientid` is populated in `<0.2%` of `normal` rows, `0%` of `mitm` rows, `0.69%` of `DoS` rows, and `7.48%` of `intrusion` rows (because MQTT `CONNECT` happens once per session).
- **Conclusion:** Per-packet **MAC/IP-level identification** is consistently possible (MAC 100%, IP ~95–100% except ARP-based MitM), whereas **MQTT `clientid`-level device identity** is only directly observable on `CONNECT` packets and must be propagated across a TCP/IP session to label subsequent packets.

### 2.4 Data Quality, Duplicates, Constant Columns, and Leakage Risks

**Verified Findings:**
- **All-missing (100% NaN) columns**: 19 columns in `DoS.csv` and `Intrusion.csv` (`frame.time_invalid`, `frame.coloring_rule.*`, `frame.comment*`, `frame.file_off`, `frame.incomplete`, `frame.interface_*`, `frame.link_nr`, `frame.md5_hash`, `mqtt.passwd*`, `mqtt.username*`, `mqtt.willmsg*`, `mqtt.willtopic*`), plus 3 more in `MitM.csv` (`mqtt.msgid`, `mqtt.sub.qos`, `mqtt.suback.qos` = 22 total).
- **Constant columns (4)**: `frame.encap_type=1`, `frame.ignored=0`, `frame.marked=0`, `frame.offset_shift=0.0`.
- **Exact full-file duplicates**: `DoS.csv`: 32,298; `MitM.csv`: 37; `Intrusion.csv`: 30.
- **Duplicates ignoring `frame.number` and `frame.time_*`**: `DoS.csv`: 76,285 (80.62%); `MitM.csv`: 98,488 (88.99%); `Intrusion.csv`: 55,964 (69.18%).
- **Leakage risks**:
  1. `frame.time_epoch`, `frame.time_relative`, and `frame.number` encode capture file and attack window directly.
  2. `ip.src`/`ip.dst`/`eth.src`/`eth.dst` map 1-to-1 to specific attacker hosts in this single-lab testbed (e.g., `b8:27:eb:73:32:30` in `mitm`).
  3. In `DoS.csv`, `mqtt-malaria/...` topics appear exclusively in `DoS` rows.
  4. Random row-level splitting leaks near-duplicate/adjacent packets across train and test sets.

---

## 3. MQTTset Deep Inspection (`MQTTset.csv`)

### 3.1 Label (`label`) and Attack Category (`Cat`) Distributions

**Verified Findings (541,071 rows, 85 columns, 0 missing values):**

| `label` | `Cat` | Row Count | % of Dataset | Timestamp Range (`Timestamp`) |
|---|---|---:|---:|---|
| `Normal` | `Normal` | 440,699 | 81.45% | `12/06/2020 01:00:01 PM` – `19/06/2020 12:59:57 AM` |
| `Anomaly` | `Flood` | 77,793 | 14.38% | `15/07/2021 04:51:38 AM` – `15/07/2021 04:54:39 AM` |
| `Anomaly` | `MalariaDoS` | 11,408 | 2.11% | `10/06/2020 10:36:59 AM` – `10/06/2020 10:39:05 AM` |
| `Anomaly` | `Bruteforce` | 4,547 | 0.84% | `22/06/2020 03:49:37 AM` – `22/06/2020 04:26:54 AM` |
| `Anomaly` | `Malformed` | 3,580 | 0.66% | `01/07/2020 12:11:16 PM` – `01/07/2020 12:17:30 PM` |
| `Anomaly` | `Slowite` | 3,044 | 0.56% | `24/07/2019 10:36:33 AM` – `24/07/2019 10:36:33 AM` |
| **Total** | **6 classes** | **541,071** | **100.00%** | — |

- `label` is a strict binary collapse of `Cat` (`Normal` -> `Normal`; the 5 attack categories -> `Anomaly`).
- **Severe class imbalance** exists across `Cat`: `Slowite` (0.56%), `Malformed` (0.66%), `Bruteforce` (0.84%), and `MalariaDoS` (2.11%) are dwarfed by `Normal` (81.45%) and `Flood` (14.38%).

### 3.2 Critical Data Leakage & Integrity Findings in `MQTTset.csv`

**Verified Findings:**
1. **`Normal` vs. `Anomaly` is trivially separated by `Protocol` and `Dst_Port`**:
   - **100% of `Normal` rows (440,699 / 440,699)** have `Protocol = 0` and `Dst_Port = 0`, with 83,283 unique `Src_IP` values (e.g., `192.215.192.168`, `193.63.192.168`) and 6 `Dst_IP` values (`0.154.10.16`, `0.155.10.16`, `0.174.10.16`, `100.73.192.168`, `0.176.10.16`, `0.180.10.16`). **Inference:** These IP strings appear byte-swapped/anonymized from a different capture source than the attack flows.
   - **100% of `Anomaly` rows (100,372 / 100,372)** have `Protocol = 6` (TCP) and `Dst_Port = 1883` (plus ephemeral return ports).
2. **Every `Cat` class has disjoint timestamps and/or disjoint IP pairs**:
   - `Flood`: `15/07/2021`, `192.168.1.73 -> 192.168.1.105:1883`
   - `Slowite`: `24/07/2019 10:36:33 AM` (all 3,044 rows in one timestamp), `10.16.120.44 -> 10.16.120.72:1883`
   - `MalariaDoS`: `10/06/2020`, `192.168.1.90 <-> 192.168.1.91`
   - `Bruteforce`: `22/06/2020`, `192.168.1.90 <-> 192.168.1.91`
   - `Malformed`: `01/07/2020`, `192.168.1.90 <-> 192.168.1.91`
   - **Consequence:** Including `Flow_ID`, `Src_IP`, `Dst_IP`, `Src_Port`, `Dst_Port`, `Protocol`, or `Timestamp` as model features causes **100% shortcut leakage** between captures rather than learning attack behavior.
3. **Constant columns (15)**: `Bwd_PSH_Flags=0`, `Fwd_URG_Flags=0`, `Bwd_URG_Flags=0`, `URG_Flag_Cnt=0`, `CWE_Flag_Count=0`, `ECE_Flag_Cnt=0`, `Fwd_Byts/b_Avg=0`, `Fwd_Pkts/b_Avg=0`, `Fwd_Blk_Rate_Avg=0`, `Bwd_Byts/b_Avg=0`, `Subflow_Bwd_Pkts=0`, `Active_Mean=0`, `Active_Std=0`, `Active_Max=0`, `Active_Min=0`.
4. **Invalid `±inf` values**: `Flow_Pkts/s` has `328` `±inf` values; `Flow_Byts/s` has `193` `±inf` values. (0 negative numeric values observed.)
5. **Duplicates**: `78` exact full-file duplicate rows; `688` duplicate rows ignoring `Flow_ID` and `Timestamp`; `383,316` duplicate `Flow_ID` rows (`157,755` unique `Flow_ID`s out of `541,071` rows).

### 3.3 Device Identity in `MQTTset.csv`
- **Verified Finding:** `MQTTset.csv` contains no MAC addresses, no `mqtt.clientid`, no `mqtt.topic`, and no `mqtt.msg`. Identification is strictly at the **IP/port/flow level** (`Src_IP`, `Dst_IP`, `Flow_ID`), and `Normal` IPs are synthetic/byte-swapped (`83,283` unique `Src_IP`s).

---

## 4. Exact Limitations to Report in the Final-Year Thesis

1. **MQTT-UAD & MQTTset cannot be merged**: They have incompatible schemas (67 packet-level Wireshark columns vs. 85 flow-level CICFlowMeter columns), different capture years (2018 vs. 2019–2021), and different label taxonomies.
2. **Neither dataset contains formal-model / TFSM-mutant / fault-localization ground truth**: Neither dataset labels which FSM state or transition was violated; they only provide packet/flow-level attack class labels.
3. **MQTTset `Normal` vs. `Anomaly` capture-artifact bias**: `Normal` (`Protocol=0, Dst_Port=0`, byte-swapped IPs) was captured/exported differently from all 5 `Anomaly` classes (`Protocol=6, Dst_Port=1883`), requiring strict exclusion of identifier/protocol/port/timestamp features and careful caveats in evaluation.
4. **MQTT-UAD `mitm` IP sparsity**: 90.09% of `mitm` packets have missing `ip.src`/`ip.dst`/`tcp.dstport` because they are Layer-2 ARP frames.
5. **MQTT-UAD `normal` MQTT sparsity & synthetic `DoS`**: Over 97% of `normal` packets in MQTT-UAD are non-MQTT background traffic, while `DoS` uses synthetic `mqtt-malaria` load-generator topics.
6. **Severe class imbalance**: `mitm` (1.35%) and `intrusion` (0.66%) in MQTT-UAD; `Slowite` (0.56%), `Malformed` (0.66%), `Bruteforce` (0.84%) in MQTTset.

