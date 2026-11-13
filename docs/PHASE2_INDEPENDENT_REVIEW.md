# Phase 2.5 — Independent Technical Audit Before Model Training

> **Audit Scope:** Independent read-only review of `features/preprocess_mqtt_uad.py`, `features/preprocess_mqttset.py`, `features/validate_phase2.py`, `datasets/processed/phase2_validation_summary.txt`, `datasets/processed/*/*_manifest.json`, sampled processed CSV rows, and the three Phase 2 reports under `docs/`.  
> **Verdict:** **READY FOR PHASE 3 (WITH CONDITIONS)** — provided the four mandatory Phase 3 modeling guards in Section 4 are enforced.

---

## 1. Verification Summary Against the 9 Audit Questions

| # | Audit Question | Verdict | Severity of Residual Risk |
|---|---|---|---|
| 1 | Does MQTT-UAD chronological splitting genuinely prevent temporal leakage? | **Yes (zero timestamp overlap), with a boundary-windowing caveat.** | **Medium** |
| 2 | What generalization claims are valid when each capture contributes to all 3 splits? | **Within-capture temporal holdout only; NOT cross-capture/unseen-session generalization.** | **High (if overstated)** |
| 3 | Does MQTTset `Flow_ID` grouping prevent related records from crossing splits? | **Prevents exact 5-tuple overlap (`0` shared `Flow_ID`s), NOT host-pair/session correlation.** | **Medium** |
| 4 | Were the `426,554` removed MQTTset rows true behavioral duplicates? | **Yes (`feature_cols + label + Cat`), but removal inverts the `Normal`/`Flood` class prior.** | **Medium** |
| 5 | Are labels and metadata correctly aligned after deduplication? | **Verified (`source_row_idx`, `label`/`Cat`/`type`, and `split` align 1-to-1 on disk).** | **None** |
| 6 | Are `-1.0` sentinels handled consistently by the feature pipeline? | **Consistent for trees; requires train-only masking/scaling before neural/linear models.** | **High (for NN/MLP)** |
| 7 | Are any retained features derived from labels, timestamps, or capture identity? | **No direct leakage; subtle capture-level sparsity (`mqtt.msgid`, `is_mqtt_port`) noted.** | **Low–Medium** |
| 8 | Are class imbalance and missing attack evidence adequately addressed? | **Documented, not yet mitigated (requires class-weighted loss & macro-F1 in Phase 3).** | **Medium** |
| 9 | Can these datasets support classification, diagnosis, and localization separately? | **Classification: Yes (caveated); Diagnosis: Partial (MQTT-UAD); Localization: No.** | **Critical (scope)** |

---

## 2. Detailed Verified Findings & Methodological Weaknesses

### 2.1 MQTT-UAD Chronological Splitting & Boundary Risk (Medium)
- **Verified:** In `datasets/processed/phase2_validation_summary.txt`, `train`, `val`, and `test` have strictly non-overlapping `frame.time_epoch` intervals within each file (`DoS.csv`, `MitM.csv`, `Intrusion.csv`).
- **Weakness:** The split boundaries have **zero temporal buffer** (e.g., `DoS.csv` `val` ends at `1522236122.430` and `test` begins at `1522236122.430`; `MitM.csv` and `Intrusion.csv` share exact boundary timestamps between `train`/`val` and `val`/`test`).
- **Impact:** Packets immediately adjacent to the boundary belong to the same ongoing TCP/MQTT burst. If Phase 3 constructs sliding-window sequences (e.g., for an LSTM) *before* splitting, windows will straddle `train`/`val`/`test`. Sequences **must** be constructed strictly within each split.

### 2.2 Valid vs. Invalid Generalization Claims
- **Valid:** Evaluating whether a model trained on the first 70% of a capture detects/classifies attacks in the final 15% of that same lab capture.
- **Invalid:** Claiming generalization to unseen IoT networks, unseen MQTT brokers, or independent capture sessions.

### 2.3 MQTTset `Flow_ID` Grouping & Deduplication Prior Shift (Medium)
- **Verified:** `mqttset_features.csv` and `mqttset_metadata.csv` show `0` `Flow_ID` overlap across `train`, `val`, and `test`, and `source_row_idx` (`0, 1, 2, 3, 5` — skipping dropped duplicate row `4`) aligns 1-to-1 with `label`, `Cat`, and `split`.
- **Weakness 1 (Ephemeral-Port Correlation):** `Flow_ID` is `Src_IP-Dst_IP-Src_Port-Dst_Port-Protocol`. Different flows from the same `Flood` burst (`192.168.1.73 -> 192.168.1.105:1883`) differ only by `Src_Port` and therefore appear in `train`, `val`, and `test`.
- **Weakness 2 (Prior Inversion):** Removing `426,554` behavioral duplicates (`420,302` from `Normal`) shifts `Normal` from `81.45%` (raw) to `17.81%` (processed) and `Flood` from `14.38%` (raw) to `67.16%` (processed). Phase 3 models must use class-weighted loss / macro-F1 and must not interpret `67.16%` as real-world attack prevalence.

### 2.4 Sentinel `-1.0` Handling in Neural / Linear Models (High for NN/MLP)
- **Verified:** `preprocess_mqtt_uad.py` fills missing values in all `27` numeric packet/MQTT columns with `-1.0` and pairs them with binary presence flags (`has_ip`, `has_tcp`, `is_mqtt_port`, `has_mqtt`, `has_mqtt_topic`, `has_mqtt_msg`).
- **Weakness:** While `-1.0` is safe for tree-based models (Random Forest, XGBoost, LightGBM), feeding raw `-1.0` into `StandardScaler`, linear models, or neural networks alongside large positive values (e.g., `mqtt.len`, `mqtt.kalive`) distorts mean/variance statistics and gradients unless masked or imputed after scaling using train-split statistics.

### 2.5 Subtle Capture-Indicator & Protocol-Presence Signals (Low–Medium)
- **Verified:** No predictive feature is derived from labels, `frame.number`, `frame.time_epoch`, `ip.*`, `eth.*`, or `mqtt.topic`/`clientid`/`msg`.
- **Subtle Risk 1:** `mqtt.msgid`, `mqtt.sub.qos`, and `mqtt.suback.qos` are `-1.0` for 100% of `MitM.csv` rows (because those 3 columns were 100% NaN in `MitM.csv`), which could subtly act as a capture indicator.
- **Subtle Risk 2:** Because `normal` in MQTT-UAD is 97%+ non-MQTT background traffic while `DoS` is 85.78% MQTT traffic, a classifier could rely heavily on `has_mqtt` / `is_mqtt_port` rather than MQTT attack behavior. Phase 3 should evaluate both the **full packet stream** and an **MQTT-only subset** (`has_mqtt == 1`) to verify genuine protocol-level discrimination.

### 2.6 Capability Separation: Classification vs. Diagnosis vs. Localization
- **Attack Detection & Classification:** Supported on MQTT-UAD (4-class `type`, within-capture) and MQTTset (6-class `Cat`, secondary caveated benchmark).
- **Attack Diagnosis:** Partially supported on MQTT-UAD (e.g., L2 ARP `mitm` via `has_ip=0` vs. MQTT `DoS` burst via `frame.time_delta` + `mqtt.msgtype`); weak on MQTTset (flow statistics only).
- **Formal Fault Localization:** **Not supported** by either MQTT-UAD or MQTTset (no TFSM state/transition mutant ground truth). Must remain anchored in `models_fsm/` + `simulation/`.

---

## 3. Claims the Project Can and Cannot Legitimately Make

- **Can Claim:**
  - Leakage-aware preprocessing with verified zero NaNs, zero `±inf`, zero `Flow_ID` overlap (MQTTset), and non-overlapping chronological holdout splits (MQTT-UAD).
  - Real-world packet/MQTT-protocol attack detection and 4-class diagnosis (`normal`, `DoS`, `mitm`, `intrusion`) on MQTT-UAD within-capture temporal splits.
  - Secondary flow-level classification on deduplicated MQTTset after removing `Protocol`/`Dst_Port`/IP/timestamp shortcut leakage.
  - Formal fault localization (mutant/transition/state attribution) on the project's TFSM + mutation-testing pipeline (`models_fsm/`, `simulation/`).
- **Cannot Claim:**
  - Cross-capture or unseen-network generalization from MQTT-UAD or MQTTset.
  - Formal FSM fault localization on MQTT-UAD or MQTTset.
  - MQTT application-layer diagnosis on MQTTset.

---

## 4. Corrections Required Before Phase 3 Model Training

1. **Sequence Isolation:** When building temporal windows (e.g., LSTM) on MQTT-UAD, construct windows *strictly inside* each split (`train`, `val`, `test`)—never across split boundaries.
2. **Sentinel & Scaling Hygiene:** Keep `-1.0` as-is for tree baselines, **or** for neural/linear models, replace `-1.0` with `0.0` (after train-only z-score normalization computed exclusively on non-missing `!= -1.0` entries) while retaining `has_mqtt*` indicators.
3. **Imbalance & Shortcut Controls:** Use class-weighted loss and report **macro-F1, per-class F1, and PR-AUC** (never accuracy alone), and run an **MQTT-only (`has_mqtt == 1`)** ablation on MQTT-UAD.
4. **No Feature-Leakage Re-Introduction:** Never join `*_metadata.csv` columns (`ip.*`, `eth.*`, `frame.time_epoch`, `Flow_ID`, `Protocol`, `Dst_Port`, `Timestamp`) into `X`.

---

## 5. Final Recommendation

**READY FOR PHASE 3 (WITH CONDITIONS)** — Phase 2 outputs are clean, aligned, and validated (`OVERALL PHASE 2 VALIDATION PASSED: True`). Model training may proceed once the four Phase 3 training-time guards in Section 4 are applied.

