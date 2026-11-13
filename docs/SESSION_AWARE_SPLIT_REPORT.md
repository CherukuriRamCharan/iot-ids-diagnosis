# Session-Aware Split Investigation and Evaluation Report

> **Core Rule:** Never randomly split individual packet/flow rows when packets from the same capture or flow can appear in both training and testing, and never manufacture artificial session IDs to create the illusion of independent-capture generalization.

---

## 1. Investigation Outcome: Do Independent Capture/Session Groups Exist?

| Dataset | Attack Class | Independent Capture Count | Independent Capture-Level Test Possible? |
|---|---|---:|---|
| **MQTT-UAD** | `DoS` | 1 (`DoS.csv`, Mar 2018) | **No** — only 1 capture contains `DoS`. |
| **MQTT-UAD** | `mitm` | 1 (`MitM.csv`, Jun 2018) | **No** — only 1 capture contains `mitm`. |
| **MQTT-UAD** | `intrusion` | 1 (`Intrusion.csv`, Sep 2018) | **No** — only 1 capture contains `intrusion`. |
| **MQTTset** | `Flood`, `MalariaDoS`, `Bruteforce`, `Malformed`, `Slowite`, `Normal` | 1 disjoint capture/IP-pair per `Cat` | **No** — each `Cat` comes from a single disjoint date/IP-pair. |

**Conclusion:** A true **cross-capture / independent-session test split** cannot be constructed from either dataset alone because each attack category appears in only one capture file/session.

---

## 2. Defensible Split Strategy Implemented for MQTT-UAD

### 2.1 Method: Contiguous Chronological 70% / 15% / 15% Block Split per Capture File
Because random row splitting would leak `76,285`–`98,488` near-duplicate packets across splits, `features/preprocess_mqtt_uad.py` sorts each capture file by `(frame.time_epoch, source_row_idx)` and assigns contiguous temporal blocks:
- First `70%` -> `train`
- Next `15%` -> `val`
- Final `15%` -> `test`

### 2.2 Verified Chronological Bounds (Zero Temporal Overlap)

| File | `train` Epoch Range | `val` Epoch Range | `test` Epoch Range |
|---|---|---|---|
| `DoS.csv` | `1522235370.391` – `1522235997.651` | `1522235997.652` – `1522236122.430` | `1522236122.430` – `1522236312.444` |
| `MitM.csv` | `1529667811.198` – `1529670982.554` | `1529670982.554` – `1529672104.766` | `1529672104.766` – `1529672517.769` |
| `Intrusion.csv` | `1535971845.729` – `1535976335.083` | `1535976335.083` – `1535976644.489` | `1535976644.489` – `1535977010.137` |

### 2.3 Verified Split Label Distribution (`286,186` Total Rows)

| File & Split | `normal` | `DoS` | `mitm` | `intrusion` | Total |
|---|---:|---:|---:|---:|---:|
| `DoS.csv` \| `train` | 33,914 | 32,323 | 0 | 0 | 66,237 |
| `DoS.csv` \| `val` | 7,489 | 6,704 | 0 | 0 | 14,193 |
| `DoS.csv` \| `test` | 7,708 | 6,487 | 0 | 0 | 14,195 |
| `MitM.csv` \| `train` | 74,937 | 0 | 2,530 | 0 | 77,467 |
| `MitM.csv` \| `val` | 15,437 | 0 | 1,163 | 0 | 16,600 |
| `MitM.csv` \| `test` | 16,439 | 0 | 162 | 0 | 16,601 |
| `Intrusion.csv` \| `train` | 54,981 | 0 | 0 | 1,644 | 56,625 |
| `Intrusion.csv` \| `val` | 11,991 | 0 | 0 | 142 | 12,133 |
| `Intrusion.csv` \| `test` | 12,023 | 0 | 0 | 112 | 12,135 |
| **Combined** | **234,919** | **45,514** | **3,855** | **1,898** | **286,186** |

- **Limitation to report:** `MitM.csv` `test` contains `162` `mitm` packets and `Intrusion.csv` `test` contains `112` `intrusion` packets, and all splits still originate from the same 3 lab captures. Therefore, **within-capture chronological evaluation** measures temporal holdout performance, **not** unseen-network/unseen-broker generalization.

---

## 3. Defensible Split Strategy Implemented for MQTTset

### 3.1 Method: Stratified `Flow_ID`-Grouped 70% / 15% / 15% Split (`SEED = 42`)
After deduplicating `541,071` raw rows down to `114,517` unique behavioral feature rows, `features/preprocess_mqttset.py` groups by `Flow_ID` (stratified by `Cat`) so that all rows sharing the same `Flow_ID` remain in the same split.

### 3.2 Verified Split Distribution (`114,517` Deduplicated Rows)

| `Cat` | `train` | `val` | `test` | Total |
|---|---:|---:|---:|---:|
| `Flood` | 56,015 | 10,870 | 10,019 | 76,904 |
| `Normal` | 14,245 | 3,070 | 3,082 | 20,397 |
| `MalariaDoS` | 5,402 | 1,776 | 3,674 | 10,852 |
| `Bruteforce` | 2,152 | 458 | 456 | 3,066 |
| `Malformed` | 1,828 | 339 | 343 | 2,510 |
| `Slowite` | 555 | 118 | 115 | 788 |
| **Total** | **80,197** | **16,631** | **17,689** | **114,517** |

- **Group Overlap Verification**: `Flow_ID` overlap across `train & val = 0`, `train & test = 0`, `val & test = 0`.
- **Limitation to report**: `Flow_ID` grouping prevents within-flow leakage, **not** capture-level leakage, because each `Cat` in `MQTTset.csv` still comes from a single capture session.

---

## 4. Recommended Alternative for Unseen-Session Generalization

To rigorously evaluate **unseen-session generalization** and **formal fault localization** in your thesis:
1. Keep your **TFSM + mutant simulation (`models_fsm/`, `simulation/`)** as the primary benchmark for **unseen-session diagnosis and fault localization** (where multiple independent network-noise sessions are generated per mutant).
2. Report **MQTT-UAD** chronological holdout results as **real packet/MQTT-protocol validation** (with the single-capture-per-attack caveat explicitly stated).
3. Report **MQTTset** `Flow_ID`-grouped results strictly as a **secondary flow-level benchmark** (with the capture-artifact and deduplication caveats explicitly stated).

