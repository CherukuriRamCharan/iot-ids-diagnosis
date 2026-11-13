# Stage 3C — Deep Learning Sequence Modeling Implementation Plan

**Project:** *Attack Diagnosis and Localization in IoT Intrusion Detection Using Formal Modeling and Deep Learning*  
**Stage:** Stage 3C — Deep Learning Sequence Modeling (LSTM, BiLSTM, GRU)  
**Primary Dataset:** MQTT-UAD (Packet-level temporal trace)  
**Benchmark Reference:** Stage 3B.6 Classical ML Baselines (`results/stage_3b6_improvements/`)

---

## 1. Executive Summary & Objective

In Stages 3B and 3B.6, classical tabular classifiers (Random Forest, ExtraTrees, HistGradientBoosting) evaluated individual packet feature vectors independently without temporal memory. While `HistGradientBoosting` achieved strong full-stream multiclass macro-F1 (**0.6975**), minority attack classes remain challenging due to packet-level ambiguity:
- `mitm`: F1 = **0.3859** (Recall: 0.2951, Precision: 0.5698)
- `intrusion`: F1 = **0.4340** (Recall: 0.5018, Precision: 0.3825)

**Stage 3C Objective:** Implement temporal sequence models (**LSTM**, **BiLSTM**, and **GRU**) that consume sliding windows of chronological packet vectors to capture temporal transitions, packet inter-arrival bursts, and protocol state shifts across time, improving minority-class attack diagnosis without introducing boundary or split leakage.

---

## 2. Existing Architecture & Reusable Components

Stage 3A established leak-free, verified sequence preparation utilities in `ids/datasets_real.py`:

| Component in `ids/datasets_real.py` | Implementation Status | Reusability in Stage 3C |
|---|:---:|---|
| `load_mqtt_uad(..., mqtt_only, binary)` | **Complete & Verified** | Loads train/val/test chronological splits with exact feature columns and metadata. |
| `MaskedStandardScaler(sentinel=-1.0)` | **Complete & Verified** | Fits exclusively on `X_train`; preserves binary indicator columns (`has_ip`, `has_tcp`, `is_mqtt_port`, `has_mqtt`, etc.) and scales valid features. |
| `compute_class_weights(y_train)` | **Complete & Verified** | Computes inverse-frequency class weights from training labels only for loss weighting. |
| `build_mqtt_uad_sequences(X, y, groups, row_ids)` | **Complete & Verified** | Constructs sliding sequence windows strictly within capture file groups and chronological splits. |
| `derive_binary_dataset(ds_mc)` | **Complete & Verified** | Fast in-memory derivation of binary tasks without reloading CSVs. |

### Zero-Leakage Guarantee
- **Zero Cross-File Windows:** `build_mqtt_uad_sequences` splits by `source_file` (`groups`). A window never combines packets from different physical capture files.
- **Zero Cross-Split Windows:** Sequence generation is invoked separately on `X_train`, `X_val`, and `X_test`.
- **Zero Normalization Leakage:** `MaskedStandardScaler` fits on `X_train` only before sequence windowing.

---

## 3. Sequence Formulation & Tensor Specifications

### Input Tensor Dimensions
For each split:
$$\mathbf{X}_{\text{seq}} \in \mathbb{R}^{N \times L \times F}, \quad \mathbf{y}_{\text{seq}} \in \mathbb{Z}^{N}$$
- $N$: Total valid sliding windows in the split.
- $L$: Sequence length / window size (default: $L = 16$ packets).
- $F$: Feature dimension ($F = 33$ predictive features, or $F = 30$ under the optimal `no_mitm_nan_cols` ablation).
- Batch dimension: $B = 128$ or $256$ during PyTorch `DataLoader` batching $\rightarrow (B, L, F)$.

### Sequence Length ($L$) & Stride ($S$)
- **Window Size ($L = 16$)**: Covers short multi-packet protocol handshakes (e.g., MQTT Connect, ConnAck, Subscribe, SubAck, Publish bursts).
- **Stride ($S$)**:
  - **Evaluation (`val` & `test`)**: $S = 1$ (complete, gapless chronological evaluation for continuous real-time intrusion monitoring).
  - **Training (`train`)**: $S = 1$ by default; stride $S = 2$ optional for memory/runtime acceleration.

### Sequence Labeling Strategy
- **Primary Contract: `"last"` (Aligned Online Detection)**
  $$y_{\text{seq}} = y[t]$$
  The sequence window $[x_{t-L+1}, \dots, x_t]$ is labeled with the label of the *latest* packet ($x_t$).
  - **Scientific Justification:** Aligns detection to current time $t$ for real-time IDS streaming. Enables a direct, fair, sample-to-sample comparison against Stage 3B.6 tabular baselines predicting $x_t$.
- **Auxiliary Mode: `"any_attack"` (Binary Burst Detection)**
  $$y_{\text{seq}} = \mathbb{I}\left(\sum_{k=0}^{L-1} y[t-k] > 0\right)$$
  Evaluated as a secondary sensitivity metric for window-level binary alert systems.

---

## 4. Model Architectures to Implement

All models will be implemented in PyTorch (`ids/models_sequence.py`) with unified parameter sizing:

```
Input (B, L, F)
      │
      ▼
Input Projection: Linear(F, D) + LayerNorm + Dropout(0.2)
      │
      ▼
Recurrent Core (LSTM / BiLSTM / GRU) [2 layers, hidden_size D, dropout 0.2]
      │
      ▼
Temporal Pooling / Final Step Representation
      │
      ▼
Classification Head: Linear(D_out, D // 2) + ReLU + Dropout(0.2) + Linear(D // 2, C)
      │
      ▼
Output Logits (B, C)
```

### 1. Unidirectional LSTM (`LSTMClassifier`)
- 2 layers, `hidden_size = 64`, unidirectional.
- Final step representation: $h_L^{(2)} \in \mathbb{R}^{64}$.
- Captures causal forward temporal dependencies in strict arrival order.

### 2. Bidirectional LSTM (`BiLSTMClassifier`)
- 2 layers, `hidden_size = 64`, bidirectional (`bidirectional=True`).
- Representation: Concatenation of forward and backward final states $[\overrightarrow{h}_L^{(2)}; \overleftarrow{h}_1^{(2)}] \in \mathbb{R}^{128}$.
- Evaluates whether contextual retrospective visibility within a short burst window improves ambiguous minority attack detection.

### 3. Gated Recurrent Unit (`GRUClassifier`)
- 2 layers, `hidden_size = 64`, unidirectional.
- Representation: $h_L^{(2)} \in \mathbb{R}^{64}$.
- More parameter-efficient than LSTM (no separate cell state $c_t$), offering faster convergence and reduced inference latency.

---

## 5. Training, Optimization & Loss Strategy

### Loss Function & Class Imbalance
- **Multiclass Tasks (4-Class)**:
  $$\mathcal{L} = \text{CrossEntropyLoss}(w_{\text{class}}), \quad w_{\text{class}} = \frac{N_{\text{train}}}{C \cdot N_c}$$
  Class weights computed strictly from `y_train_seq` to heavily penalize misclassification of minority classes (`mitm` and `intrusion`).
- **Binary Tasks**:
  $$\mathcal{L} = \text{BCEWithLogitsLoss}(\text{pos\_weight} = w_{\text{attack}})$$

### Optimizer & Learning Rate Schedule
- **Optimizer**: `AdamW(lr=1e-3, weight_decay=1e-4)`.
- **Scheduler**: `ReduceLROnPlateau(mode='max', factor=0.5, patience=2)` monitoring validation Macro-F1.
- **Gradient Clipping**: `torch.nn.utils.clip_grad_norm_(max_norm=1.0)` to eliminate exploding gradients in recurrent backpropagation through time (BPTT).
- **Early Stopping**: Patience = 6 epochs based on validation Macro-F1. Maximum epochs = 25.

---

## 6. Experimental Tasks & Benchmark Comparison

Stage 3C evaluates the exact same 4 tasks as Stage 3B.6 on MQTT-UAD:

1. `mqtt_uad_full_multiclass`: Full Stream 4-Class (`normal`, `DoS`, `mitm`, `intrusion`)
2. `mqtt_uad_full_binary`: Full Stream Binary (`normal`, `attack`)
3. `mqtt_uad_mqtt_only_multiclass`: MQTT-Only 4-Class
4. `mqtt_uad_mqtt_only_binary`: MQTT-Only Binary

### Comparative Benchmark (Stage 3B.6 Baselines)

| Task | Stage 3B.6 Best Model | Stage 3B.6 Macro-F1 | Stage 3B.6 Balanced Acc | Stage 3B.6 Minority F1 | Stage 3C Hypothesis |
|---|---|:---:|:---:|:---:|---|
| **Full Multiclass** | `HistGB_tuned` | **0.6975** | 0.6911 | `mitm`: 0.3859<br>`intrusion`: 0.4340 | Sequence modeling should leverage inter-packet arrival jitter (`frame.time_delta`) to raise `mitm` and `intrusion` F1 beyond 0.50. |
| **Full Binary** | `HistGB_tuned` | **0.9833** | 0.9818 | Attack F1: 0.9718 | Sequence context should maintain >0.98 Macro-F1 while further suppressing transient false positives. |
| **MQTT-Only Multiclass** | `RF_train_oversampled` | **0.6831** | 0.6782 | `mitm`: 0.1333<br>`intrusion`: 0.6460 | Tests whether sequence temporal structure can extract signal from the 13-sample MITM test support. |
| **MQTT-Only Binary** | `RF_tuned` | **0.9783** | 0.9871 | Attack F1: 0.9956 | Rapid sequence verification on pure MQTT streams. |

### Secondary Dataset Strategy (MQTTset)
- **Decision: Omit from primary Stage 3C sequence experiments.**
- *Scientific Justification:* The Stage 3B.5 audit confirmed MQTTset samples are flow-aggregated statistics rather than consecutive packet streams, and single-file capture artifacts dominate. Sequences are only physically meaningful on packet-level temporal traces (MQTT-UAD).

---

## 7. Checkpointing, Artifacts & Reproducibility

Following the verified Stage 3B.6 standard:
- **Directory**: `results/stage_3c_sequences/checkpoints/`
- **Naming**: `{task_id}__{model_family}__seed{seed}.json`
- **Model Weights**: `results/stage_3c_sequences/models/{task_id}__{model_family}__seed{seed}_best.pt`
- **Resume Capability**: `--resume` skips valid completed JSON checkpoints atomically.
- **Reporting**: Generates `docs/STAGE3C_SEQUENCE_MODELING_REPORT.md` and `results/stage_3c_sequences/model_comparison.csv`.

---

## 8. Smallest Sensible Initial Experiment (Smoke Test)

Before launching full 5-seed multi-epoch runs across all 4 tasks:
- **Scope**:
  - Task: `mqtt_uad_full_multiclass`
  - Model: `LSTM`
  - Seed: 42
  - Fast mode: Subsample 5,000 training windows, 1,000 validation windows, 1 epoch.
- **Success Criteria**:
  1. `build_mqtt_uad_sequences` executes without error and yields $(N, 16, 33)$ tensors.
  2. Forward pass, loss calculation, backward pass, and optimizer step succeed.
  3. Metric computation outputs Macro-F1, Balanced Acc, and Confusion Matrix.
  4. Best checkpoint is saved and reloadable.
  5. Wall-clock execution time: **$< 30$ seconds**.

---

## 9. Stage 3C Deliverables & Next Action

### Files to Create in Stage 3C
1. `ids/models_sequence.py`: PyTorch model definitions (`LSTMClassifier`, `BiLSTMClassifier`, `GRUClassifier`).
2. `ids/train_sequences.py`: Training engine with validation Macro-F1 selection, class-weighted loss, threshold tuning, atomic checkpointing, and `--resume`.
3. `tests/test_stage3c_sequences.py`: Fast unit tests on small synthetic sequence fixtures testing tensor shapes, zero split boundary crossing, and checkpoint validation.
4. `docs/STAGE3C_SEQUENCE_MODELING_REPORT.md`: Comprehensive results report comparing Stage 3C against Stage 3B.6.

### Immediate Next Step
When approved, implement the unit tests and model architectures, and run the fast smoke test:
```cmd
python -m unittest tests.test_stage3c_sequences -v
```

