# Stage 3B Classical Machine Learning Baseline Report

> **Seeds:** `[42, 43, 44, 45, 46]` | **Primary Selection Metric:** `Macro-F1`
> **Scope Note:** MQTT-UAD (within-capture chronological holdout) and MQTTset (secondary Flow_ID-grouped deduplicated benchmark) are evaluated strictly separately. Classification metrics do **not** prove cross-capture generalization, attack diagnosis, or formal fault localization.

## 1. Experimental Setup & Safeguards
- **Models:** `RandomForestClassifier` (`class_weight='balanced'`, `n_estimators=100`, `max_depth=20`) and `DecisionTreeClassifier` (`class_weight='balanced'`, `max_depth` tuned on validation `macro_f1`).
- **Leakage Controls:** Loaded exclusively via Stage 3A `ids/datasets_real.py` (`assert_no_leakage_columns`). No metadata, timestamps, IPs, MACs, ports, topics, or client IDs were used.

## 2. Model Comparison Summary (Mean ± Std Across Seeds)

| Benchmark | Task | Model | Macro-F1 | Accuracy | Macro-P | Macro-R | Weighted-F1 | Train (s) | Infer (ms/1k) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| MQTT-UAD | MQTT-UAD (Full Stream, 4-Class) | RandomForest | 0.6098 ± 0.0009 | 0.9696 ± 0.0003 | 0.5712 | 0.8302 | 0.9778 | 1.78 | 1.714 |
| MQTT-UAD | MQTT-UAD (Full Stream, 4-Class) | DecisionTree | 0.6088 ± 0.0000 | 0.9746 ± 0.0000 | 0.5770 | 0.7345 | 0.9797 | 3.48 | 0.102 |
| MQTT-UAD | MQTT-UAD (Full Stream, Binary) | RandomForest | 0.9735 ± 0.0002 | 0.9857 ± 0.0001 | 0.9666 | 0.9807 | 0.9858 | 2.39 | 1.829 |
| MQTT-UAD | MQTT-UAD (Full Stream, Binary) | DecisionTree | 0.9734 ± 0.0003 | 0.9857 ± 0.0001 | 0.9683 | 0.9787 | 0.9858 | 3.98 | 0.105 |
| MQTT-UAD | MQTT-UAD (MQTT-Only, 4-Class) | RandomForest | 0.6899 ± 0.0024 | 0.9906 ± 0.0002 | 0.7623 | 0.7015 | 0.9902 | 0.47 | 6.839 |
| MQTT-UAD | MQTT-UAD (MQTT-Only, 4-Class) | DecisionTree | 0.6756 ± 0.0036 | 0.9875 ± 0.0022 | 0.7548 | 0.6753 | 0.9870 | 0.31 | 0.097 |
| MQTT-UAD | MQTT-UAD (MQTT-Only, Binary) | RandomForest | 0.9776 ± 0.0023 | 0.9918 ± 0.0008 | 0.9718 | 0.9835 | 0.9919 | 0.43 | 6.896 |
| MQTT-UAD | MQTT-UAD (MQTT-Only, Binary) | DecisionTree | 0.9749 ± 0.0044 | 0.9907 ± 0.0017 | 0.9612 | 0.9899 | 0.9908 | 0.29 | 0.105 |
| MQTTset | MQTTset (Secondary Flow Benchmark, 6-Class) | RandomForest | 0.9834 ± 0.0003 | 0.9969 ± 0.0001 | 0.9791 | 0.9879 | 0.9970 | 0.87 | 3.866 |
| MQTTset | MQTTset (Secondary Flow Benchmark, 6-Class) | DecisionTree | 0.9870 ± 0.0008 | 0.9976 ± 0.0002 | 0.9845 | 0.9896 | 0.9976 | 3.49 | 0.118 |
| MQTTset | MQTTset (Secondary Flow Benchmark, Binary) | RandomForest | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.67 | 3.018 |
| MQTTset | MQTTset (Secondary Flow Benchmark, Binary) | DecisionTree | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.92 | 0.076 |

## 3. Per-Class Test Metrics & Absent-Class Limitations

### MQTT-UAD (Full Stream, 4-Class) — RandomForest
- Train/Val/Test rows: `200,329` / `42,926` / `42,931` | Features: `33`

| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 36,170 | 0.9959 ± 0.0000 | 0.9804 ± 0.0003 | 0.9881 ± 0.0002 |
| `DoS` | 6,487 | 0.9825 ± 0.0007 | 0.9212 ± 0.0016 | 0.9509 ± 0.0008 |
| `mitm` | 162 | 0.1653 ± 0.0041 | 0.6049 ± 0.0055 | 0.2597 ± 0.0054 |
| `intrusion` | 112 | 0.1411 ± 0.0021 | 0.8143 ± 0.0182 | 0.2405 ± 0.0036 |

### MQTT-UAD (Full Stream, 4-Class) — DecisionTree
- Train/Val/Test rows: `200,329` / `42,926` / `42,931` | Features: `33`

| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 36,170 | 0.9929 ± 0.0000 | 0.9871 ± 0.0000 | 0.9900 ± 0.0000 |
| `DoS` | 6,487 | 0.9831 ± 0.0001 | 0.9246 ± 0.0000 | 0.9530 ± 0.0000 |
| `mitm` | 162 | 0.2086 ± 0.0000 | 0.4815 ± 0.0000 | 0.2910 ± 0.0000 |
| `intrusion` | 112 | 0.1234 ± 0.0001 | 0.5446 ± 0.0000 | 0.2012 ± 0.0002 |

### MQTT-UAD (Full Stream, Binary) — RandomForest
- Train/Val/Test rows: `200,329` / `42,926` / `42,931` | Features: `33`

| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 36,170 | 0.9950 ± 0.0001 | 0.9880 ± 0.0001 | 0.9915 ± 0.0001 |
| `attack` | 6,761 | 0.9382 ± 0.0005 | 0.9735 ± 0.0004 | 0.9555 ± 0.0003 |

### MQTT-UAD (Full Stream, Binary) — DecisionTree
- Train/Val/Test rows: `200,329` / `42,926` / `42,931` | Features: `33`

| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 36,170 | 0.9941 ± 0.0005 | 0.9889 ± 0.0003 | 0.9915 ± 0.0001 |
| `attack` | 6,761 | 0.9425 ± 0.0015 | 0.9685 ± 0.0027 | 0.9553 ± 0.0006 |

### MQTT-UAD (MQTT-Only, 4-Class) — RandomForest
- Train/Val/Test rows: `34,137` / `7,562` / `6,273` | Features: `33`

| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 626 | 0.9566 ± 0.0010 | 0.9505 ± 0.0014 | 0.9535 ± 0.0010 |
| `DoS` | 5,585 | 0.9991 ± 0.0001 | 0.9991 ± 0.0001 | 0.9991 ± 0.0001 |
| `mitm` | 13 | 0.5000 ± 0.0000 | 0.0769 ± 0.0000 | 0.1333 ± 0.0000 |
| `intrusion` | 49 | 0.5933 ± 0.0112 | 0.7796 ± 0.0082 | 0.6738 ± 0.0086 |

### MQTT-UAD (MQTT-Only, 4-Class) — DecisionTree
- Train/Val/Test rows: `34,137` / `7,562` / `6,273` | Features: `33`

| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 626 | 0.9272 ± 0.0192 | 0.9546 ± 0.0008 | 0.9406 ± 0.0101 |
| `DoS` | 5,585 | 0.9986 ± 0.0001 | 0.9960 ± 0.0024 | 0.9973 ± 0.0012 |
| `mitm` | 13 | 0.5000 ± 0.0000 | 0.0769 ± 0.0000 | 0.1333 ± 0.0000 |
| `intrusion` | 49 | 0.5936 ± 0.0052 | 0.6735 ± 0.0000 | 0.6310 ± 0.0030 |

### MQTT-UAD (MQTT-Only, Binary) — RandomForest
- Train/Val/Test rows: `34,137` / `7,562` / `6,273` | Features: `33`

| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 626 | 0.9466 ± 0.0006 | 0.9732 ± 0.0083 | 0.9597 ± 0.0041 |
| `attack` | 5,647 | 0.9970 ± 0.0009 | 0.9939 ± 0.0001 | 0.9955 ± 0.0004 |

### MQTT-UAD (MQTT-Only, Binary) — DecisionTree
- Train/Val/Test rows: `34,137` / `7,562` / `6,273` | Features: `33`

| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 626 | 0.9236 ± 0.0145 | 0.9888 ± 0.0000 | 0.9550 ± 0.0079 |
| `attack` | 5,647 | 0.9988 ± 0.0000 | 0.9909 ± 0.0019 | 0.9948 ± 0.0010 |

### MQTTset (Secondary Flow Benchmark, 6-Class) — RandomForest
- Train/Val/Test rows: `80,197` / `16,631` / `17,689` | Features: `62`

| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `Normal` | 3,082 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| `Flood` | 10,019 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| `MalariaDoS` | 3,674 | 0.9984 ± 0.0000 | 0.9911 ± 0.0002 | 0.9947 ± 0.0001 |
| `Bruteforce` | 456 | 0.9647 ± 0.0021 | 0.9842 ± 0.0009 | 0.9744 ± 0.0015 |
| `Malformed` | 343 | 0.9115 ± 0.0017 | 0.9609 ± 0.0014 | 0.9356 ± 0.0014 |
| `Slowite` | 115 | 1.0000 ± 0.0000 | 0.9913 ± 0.0000 | 0.9956 ± 0.0000 |

### MQTTset (Secondary Flow Benchmark, 6-Class) — DecisionTree
- Train/Val/Test rows: `80,197` / `16,631` / `17,689` | Features: `62`

| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `Normal` | 3,082 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| `Flood` | 10,019 | 0.9996 ± 0.0001 | 1.0000 ± 0.0000 | 0.9998 ± 0.0001 |
| `MalariaDoS` | 3,674 | 0.9991 ± 0.0001 | 0.9942 ± 0.0006 | 0.9967 ± 0.0003 |
| `Bruteforce` | 456 | 0.9667 ± 0.0045 | 0.9807 ± 0.0016 | 0.9737 ± 0.0027 |
| `Malformed` | 343 | 0.9413 ± 0.0022 | 0.9627 ± 0.0029 | 0.9519 ± 0.0020 |
| `Slowite` | 115 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

### MQTTset (Secondary Flow Benchmark, Binary) — RandomForest
- Train/Val/Test rows: `80,197` / `16,631` / `17,689` | Features: `62`

| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `Normal` | 3,082 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| `Anomaly` | 14,607 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

### MQTTset (Secondary Flow Benchmark, Binary) — DecisionTree
- Train/Val/Test rows: `80,197` / `16,631` / `17,689` | Features: `62`

| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `Normal` | 3,082 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| `Anomaly` | 14,607 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

## 4. Key Limitations
1. **MQTT-UAD:** Chronological splits are within single capture files; results do not establish unseen-capture generalization.
2. **MQTTset:** Secondary deduplicated flow benchmark (`426,554` duplicates removed); `Flow_ID` grouping does not remove single-capture-per-`Cat` artifacts.
3. **Scope:** Neither dataset contains formal TFSM fault-localization ground truth.
