# Stage 3B.6 — Controlled Baseline Improvement Report (MQTT-UAD)

> **Seeds:** `[42, 43, 44, 45, 46]` | **Primary Selection Metric:** Validation `Macro-F1`
> **Immutable Reference:** `results/stage_3b_baselines/` (unmodified). All evaluations remain within-capture chronological holdout on MQTT-UAD.

## 1. Original Stage 3B vs. Improved Stage 3B.6 Comparison

| Task | Model | Macro-F1 (mean ± std) | Balanced Acc | Accuracy | `mitm` F1 | `intrusion` F1 | Train (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| MQTT-UAD (Full Stream, 4-Class) | [Stage 3B] RandomForest | 0.6098 ± 0.0009 | 0.8302 | 0.9696 | 0.2597 | 0.2405 | 1.78 |
| MQTT-UAD (Full Stream, 4-Class) | [Stage 3B] DecisionTree | 0.6088 ± 0.0000 | 0.7345 | 0.9746 | 0.2910 | 0.2012 | 3.48 |
| MQTT-UAD (Full Stream, Binary) | [Stage 3B] RandomForest | 0.9735 ± 0.0002 | 0.9807 | 0.9857 | — | — | 2.39 |
| MQTT-UAD (Full Stream, Binary) | [Stage 3B] DecisionTree | 0.9734 ± 0.0003 | 0.9787 | 0.9857 | — | — | 3.98 |
| MQTT-UAD (MQTT-Only, 4-Class) | [Stage 3B] RandomForest | 0.6899 ± 0.0024 | 0.7015 | 0.9906 | 0.1333 | 0.6738 | 0.47 |
| MQTT-UAD (MQTT-Only, 4-Class) | [Stage 3B] DecisionTree | 0.6756 ± 0.0036 | 0.6753 | 0.9875 | 0.1333 | 0.6310 | 0.31 |
| MQTT-UAD (MQTT-Only, Binary) | [Stage 3B] RandomForest | 0.9776 ± 0.0023 | 0.9835 | 0.9918 | — | — | 0.43 |
| MQTT-UAD (MQTT-Only, Binary) | [Stage 3B] DecisionTree | 0.9749 ± 0.0044 | 0.9899 | 0.9907 | — | — | 0.29 |
| MQTT-UAD (Full Stream, 4-Class) | [Stage 3B.6] RF_tuned | 0.6889 ± 0.0019 | 0.6985 ± 0.0016 | 0.9873 | 0.3830 | 0.4048 | 754.43 |
| MQTT-UAD (Full Stream, 4-Class) | [Stage 3B.6] ExtraTrees_tuned | 0.6865 ± 0.0013 | 0.6341 ± 0.0007 | 0.9889 | 0.3826 | 0.3917 | 664.63 |
| MQTT-UAD (Full Stream, 4-Class) | [Stage 3B.6] HistGB_tuned | 0.6975 ± 0.0009 | 0.6911 ± 0.0050 | 0.9880 | 0.3859 | 0.4340 | 582.50 |
| MQTT-UAD (Full Stream, 4-Class) | [Stage 3B.6] RF_train_oversampled | 0.6412 ± 0.0046 | 0.7999 ± 0.0115 | 0.9777 | 0.3527 | 0.2672 | 157.60 |
| MQTT-UAD (Full Stream, Binary) | [Stage 3B.6] RF_tuned | 0.9818 ± 0.0004 | 0.9801 ± 0.0005 | 0.9904 | — | — | 9077.16 |
| MQTT-UAD (Full Stream, Binary) | [Stage 3B.6] ExtraTrees_tuned | 0.9824 ± 0.0002 | 0.9812 ± 0.0008 | 0.9907 | — | — | 474.33 |
| MQTT-UAD (Full Stream, Binary) | [Stage 3B.6] HistGB_tuned | 0.9833 ± 0.0000 | 0.9818 ± 0.0001 | 0.9912 | — | — | 69.34 |
| MQTT-UAD (Full Stream, Binary) | [Stage 3B.6] RF_train_oversampled | 0.9818 ± 0.0003 | 0.9803 ± 0.0002 | 0.9904 | — | — | 1339.78 |
| MQTT-UAD (MQTT-Only, 4-Class) | [Stage 3B.6] RF_tuned | 0.6701 ± 0.0023 | 0.6669 ± 0.0025 | 0.9915 | 0.0000 | 0.7227 | 30.58 |
| MQTT-UAD (MQTT-Only, 4-Class) | [Stage 3B.6] ExtraTrees_tuned | 0.6651 ± 0.0152 | 0.6526 ± 0.0112 | 0.9912 | 0.0267 | 0.6754 | 27.85 |
| MQTT-UAD (MQTT-Only, 4-Class) | [Stage 3B.6] HistGB_tuned | 0.6701 ± 0.0038 | 0.6665 ± 0.0017 | 0.9916 | 0.0000 | 0.7218 | 16.33 |
| MQTT-UAD (MQTT-Only, 4-Class) | [Stage 3B.6] RF_train_oversampled | 0.6831 ± 0.0016 | 0.6782 ± 0.0025 | 0.9903 | 0.1333 | 0.6460 | 16.84 |
| MQTT-UAD (MQTT-Only, Binary) | [Stage 3B.6] RF_tuned | 0.9783 ± 0.0005 | 0.9871 ± 0.0013 | 0.9921 | — | — | 30.47 |
| MQTT-UAD (MQTT-Only, Binary) | [Stage 3B.6] ExtraTrees_tuned | 0.9780 ± 0.0003 | 0.9841 ± 0.0006 | 0.9920 | — | — | 12.04 |
| MQTT-UAD (MQTT-Only, Binary) | [Stage 3B.6] HistGB_tuned | 0.9780 ± 0.0005 | 0.9907 ± 0.0008 | 0.9919 | — | — | 14.60 |
| MQTT-UAD (MQTT-Only, Binary) | [Stage 3B.6] RF_train_oversampled | 0.9763 ± 0.0017 | 0.9799 ± 0.0030 | 0.9914 | — | — | 10.25 |

> **Note on Reported Training Times:** For `[Stage 3B.6] RF_tuned` on Full Stream Binary, Seed 44 ran overnight and recorded a wall-clock elapsed time of 41,561.36s due to system suspension; the remaining 4 seeds recorded elapsed wall-clock times averaging 456.12s (Seed 42: 429.85s, Seed 43: 426.35s, Seed 45: 482.88s, Seed 46: 485.39s). Actual CPU process time was not recorded.

## 2. Feature Ablation on MQTT-UAD Full-Stream 4-Class (`HistGradientBoosting`, Seed 42)

| Subset | # Features | Macro-F1 | Balanced Acc | `DoS` F1 | `mitm` F1 (Recall) | `intrusion` F1 (Recall) |
|---|---:|---:|---:|---:|---:|---:|
| `all_33_features` | 33 | 0.6886 | 0.6963 | 0.9759 | 0.3817 (0.3086) | 0.4028 (0.5089) |
| `no_layer_flags` | 30 | 0.6805 | 0.6185 | 0.9579 | 0.3843 (0.2716) | 0.3899 (0.2768) |
| `no_mitm_nan_cols` | 30 | 0.6937 | 0.6939 | 0.9750 | 0.3817 (0.3086) | 0.4242 (0.5000) |
| `packet_and_layer_only` | 7 | 0.6854 | 0.6659 | 0.9666 | 0.3843 (0.2716) | 0.3984 (0.4375) |
| `mqtt_features_only` | 26 | 0.5281 | 0.5095 | 0.9184 | 0.0000 (0.0000) | 0.2100 (0.1875) |

## 3. Validation-Only Binary Threshold Tuning Summary

| Task | Model | Selected `tau*` (mean) | Default `0.50` Macro-F1 | Tuned `tau*` Macro-F1 | Attack P / R (`0.50` -> `tau*`) |
|---|---|---:|---:|---:|---|
| MQTT-UAD (Full Stream, Binary) | RF_tuned | 0.65 | 0.9818 | 0.9804 | 0.9737/0.9650 -> 0.9867/0.9477 |
| MQTT-UAD (Full Stream, Binary) | ExtraTrees_tuned | 0.56 | 0.9824 | 0.9822 | 0.9735/0.9673 -> 0.9765/0.9635 |
| MQTT-UAD (Full Stream, Binary) | HistGB_tuned | 0.65 | 0.9833 | 0.9820 | 0.9757/0.9680 -> 0.9853/0.9545 |
| MQTT-UAD (Full Stream, Binary) | RF_train_oversampled | 0.65 | 0.9818 | 0.9806 | 0.9730/0.9656 -> 0.9866/0.9485 |
| MQTT-UAD (MQTT-Only, Binary) | RF_tuned | 0.63 | 0.9783 | 0.9782 | 0.9979/0.9933 -> 0.9986/0.9925 |
| MQTT-UAD (MQTT-Only, Binary) | ExtraTrees_tuned | 0.84 | 0.9780 | 0.9752 | 0.9971/0.9940 -> 0.9980/0.9918 |
| MQTT-UAD (MQTT-Only, Binary) | HistGB_tuned | 0.67 | 0.9780 | 0.9775 | 0.9988/0.9922 -> 0.9992/0.9916 |
| MQTT-UAD (MQTT-Only, Binary) | RF_train_oversampled | 0.67 | 0.9763 | 0.9788 | 0.9962/0.9943 -> 0.9977/0.9937 |

## 4. Per-Class Test Metrics (Stage 3B.6 Models)

### MQTT-UAD (Full Stream, 4-Class) — RF_tuned
| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 36,170 | 0.9925 ± 0.0001 | 0.9959 ± 0.0001 | 0.9942 ± 0.0001 |
| `DoS` | 6,487 | 0.9833 ± 0.0011 | 0.9642 ± 0.0011 | 0.9736 ± 0.0004 |
| `mitm` | 162 | 0.5296 ± 0.0107 | 0.3000 ± 0.0030 | 0.3830 ± 0.0036 |
| `intrusion` | 112 | 0.3261 ± 0.0122 | 0.5339 ± 0.0067 | 0.4048 ± 0.0104 |

### MQTT-UAD (Full Stream, 4-Class) — ExtraTrees_tuned
| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 36,170 | 0.9935 ± 0.0000 | 0.9952 ± 0.0001 | 0.9944 ± 0.0000 |
| `DoS` | 6,487 | 0.9706 ± 0.0002 | 0.9840 ± 0.0003 | 0.9773 ± 0.0001 |
| `mitm` | 162 | 0.5719 ± 0.0190 | 0.2877 ± 0.0030 | 0.3826 ± 0.0020 |
| `intrusion` | 112 | 0.7160 ± 0.0139 | 0.2696 ± 0.0036 | 0.3917 ± 0.0034 |

### MQTT-UAD (Full Stream, 4-Class) — HistGB_tuned
| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 36,170 | 0.9930 ± 0.0003 | 0.9954 ± 0.0005 | 0.9942 ± 0.0001 |
| `DoS` | 6,487 | 0.9793 ± 0.0012 | 0.9721 ± 0.0019 | 0.9757 ± 0.0006 |
| `mitm` | 162 | 0.5698 ± 0.0627 | 0.2951 ± 0.0197 | 0.3859 ± 0.0018 |
| `intrusion` | 112 | 0.3825 ± 0.0069 | 0.5018 ± 0.0036 | 0.4340 ± 0.0041 |

### MQTT-UAD (Full Stream, 4-Class) — RF_train_oversampled
| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 36,170 | 0.9943 ± 0.0002 | 0.9909 ± 0.0007 | 0.9926 ± 0.0004 |
| `DoS` | 6,487 | 0.9875 ± 0.0013 | 0.9193 ± 0.0037 | 0.9521 ± 0.0019 |
| `mitm` | 162 | 0.2843 ± 0.0238 | 0.4679 ± 0.0120 | 0.3527 ± 0.0158 |
| `intrusion` | 112 | 0.1597 ± 0.0016 | 0.8214 ± 0.0600 | 0.2672 ± 0.0046 |

### MQTT-UAD (Full Stream, Binary) — RF_tuned
| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 36,170 | 0.9935 ± 0.0002 | 0.9951 ± 0.0002 | 0.9943 ± 0.0001 |
| `attack` | 6,761 | 0.9737 ± 0.0009 | 0.9650 ± 0.0008 | 0.9693 ± 0.0007 |

### MQTT-UAD (Full Stream, Binary) — ExtraTrees_tuned
| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 36,170 | 0.9939 ± 0.0004 | 0.9951 ± 0.0003 | 0.9945 ± 0.0001 |
| `attack` | 6,761 | 0.9735 ± 0.0017 | 0.9673 ± 0.0020 | 0.9704 ± 0.0004 |

### MQTT-UAD (Full Stream, Binary) — HistGB_tuned
| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 36,170 | 0.9940 ± 0.0000 | 0.9955 ± 0.0000 | 0.9948 ± 0.0000 |
| `attack` | 6,761 | 0.9757 ± 0.0002 | 0.9680 ± 0.0001 | 0.9718 ± 0.0001 |

### MQTT-UAD (Full Stream, Binary) — RF_train_oversampled
| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 36,170 | 0.9936 ± 0.0001 | 0.9950 ± 0.0002 | 0.9943 ± 0.0001 |
| `attack` | 6,761 | 0.9730 ± 0.0012 | 0.9656 ± 0.0004 | 0.9693 ± 0.0006 |

### MQTT-UAD (MQTT-Only, 4-Class) — RF_tuned
| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 626 | 0.9519 ± 0.0007 | 0.9665 ± 0.0000 | 0.9591 ± 0.0004 |
| `DoS` | 5,585 | 0.9982 ± 0.0000 | 0.9991 ± 0.0000 | 0.9987 ± 0.0000 |
| `mitm` | 13 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| `intrusion` | 49 | 0.7446 ± 0.0084 | 0.7020 ± 0.0100 | 0.7227 ± 0.0088 |

### MQTT-UAD (MQTT-Only, 4-Class) — ExtraTrees_tuned
| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 626 | 0.9485 ± 0.0022 | 0.9716 ± 0.0012 | 0.9599 ± 0.0007 |
| `DoS` | 5,585 | 0.9981 ± 0.0001 | 0.9989 ± 0.0000 | 0.9985 ± 0.0000 |
| `mitm` | 13 | 0.1000 ± 0.2000 | 0.0154 ± 0.0308 | 0.0267 ± 0.0533 |
| `intrusion` | 49 | 0.7358 ± 0.0079 | 0.6245 ± 0.0163 | 0.6754 ± 0.0084 |

### MQTT-UAD (MQTT-Only, 4-Class) — HistGB_tuned
| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 626 | 0.9508 ± 0.0007 | 0.9690 ± 0.0034 | 0.9598 ± 0.0017 |
| `DoS` | 5,585 | 0.9982 ± 0.0001 | 0.9990 ± 0.0001 | 0.9986 ± 0.0001 |
| `mitm` | 13 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| `intrusion` | 49 | 0.7484 ± 0.0337 | 0.6980 ± 0.0082 | 0.7218 ± 0.0136 |

### MQTT-UAD (MQTT-Only, 4-Class) — RF_train_oversampled
| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 626 | 0.9537 ± 0.0000 | 0.9553 ± 0.0000 | 0.9545 ± 0.0000 |
| `DoS` | 5,585 | 0.9983 ± 0.0001 | 0.9991 ± 0.0000 | 0.9987 ± 0.0000 |
| `mitm` | 13 | 0.5000 ± 0.0000 | 0.0769 ± 0.0000 | 0.1333 ± 0.0000 |
| `intrusion` | 49 | 0.6139 ± 0.0035 | 0.6816 ± 0.0100 | 0.6460 ± 0.0064 |

### MQTT-UAD (MQTT-Only, Binary) — RF_tuned
| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 626 | 0.9420 ± 0.0032 | 0.9808 ± 0.0029 | 0.9610 ± 0.0009 |
| `attack` | 5,647 | 0.9979 ± 0.0003 | 0.9933 ± 0.0004 | 0.9956 ± 0.0001 |

### MQTT-UAD (MQTT-Only, Binary) — ExtraTrees_tuned
| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 626 | 0.9472 ± 0.0001 | 0.9741 ± 0.0012 | 0.9605 ± 0.0006 |
| `attack` | 5,647 | 0.9971 ± 0.0001 | 0.9940 ± 0.0000 | 0.9955 ± 0.0001 |

### MQTT-UAD (MQTT-Only, Binary) — HistGB_tuned
| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 626 | 0.9334 ± 0.0025 | 0.9891 ± 0.0019 | 0.9604 ± 0.0009 |
| `attack` | 5,647 | 0.9988 ± 0.0002 | 0.9922 ± 0.0003 | 0.9955 ± 0.0001 |

### MQTT-UAD (MQTT-Only, Binary) — RF_train_oversampled
| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |
|---|---:|---:|---:|---:|
| `normal` | 626 | 0.9494 ± 0.0006 | 0.9655 ± 0.0060 | 0.9574 ± 0.0030 |
| `attack` | 5,647 | 0.9962 ± 0.0007 | 0.9943 ± 0.0001 | 0.9952 ± 0.0003 |

## 5. Limitations & Scientific Scope
1. **Within-capture chronological evaluation only:** Does not establish unseen-capture or unseen-broker generalization.
2. **MQTT-Only `mitm` test support (`13` samples):** Too small for statistically reliable `mitm` evaluation in `MQTT-Only` mode.
3. **No formal fault-localization labels:** Classification improvements do not imply TFSM fault localization.
