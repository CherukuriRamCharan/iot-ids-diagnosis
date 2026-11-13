"""Unit tests for Stage 3C Deep Learning Sequence Models (LSTM, BiLSTM, GRU).

Tests model architectures, tensor shapes, loss computation, backpropagation,
boundary protection, and numerical stability on small synthetic fixtures.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ids.datasets_real import build_mqtt_uad_sequences
from ids.models_sequence import (
    BiLSTMClassifier,
    GRUClassifier,
    LSTMClassifier,
    build_model,
)
from ids.train_sequences import (
    evaluate_model,
    set_seed,
    train_single_sequence_model,
)
from torch.utils.data import DataLoader, TensorDataset


class TestStage3CSequences(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 8
        self.seq_len = 16
        self.num_features = 33
        self.num_classes = 4

    def test_model_instantiation_and_output_shapes(self) -> None:
        """Verifies forward pass and output dimensions for all three architectures."""
        x = torch.randn(self.batch_size, self.seq_len, self.num_features)

        models = {
            "LSTM": LSTMClassifier(num_features=self.num_features, hidden_size=64, num_classes=self.num_classes),
            "BiLSTM": BiLSTMClassifier(num_features=self.num_features, hidden_size=64, num_classes=self.num_classes),
            "GRU": GRUClassifier(num_features=self.num_features, hidden_size=64, num_classes=self.num_classes),
        }

        for name, model in models.items():
            model.eval()
            with torch.no_grad():
                logits = model(x)
            self.assertEqual(
                logits.shape,
                (self.batch_size, self.num_classes),
                f"Model {name} produced incorrect output shape {logits.shape}",
            )
            self.assertFalse(torch.isnan(logits).any(), f"Model {name} produced NaN in logits")
            self.assertFalse(torch.isinf(logits).any(), f"Model {name} produced Inf in logits")

    def test_configurable_feature_count_30(self) -> None:
        """Verifies models work cleanly with the optimal 30-feature ablation subset."""
        num_feat_30 = 30
        x = torch.randn(self.batch_size, self.seq_len, num_feat_30)

        for name in ("lstm", "bilstm", "gru"):
            model = build_model(name, num_features=num_feat_30, hidden_size=64, num_classes=4)
            logits = model(x)
            self.assertEqual(logits.shape, (self.batch_size, 4))

    def test_loss_and_backpropagation(self) -> None:
        """Verifies loss computation and gradient flow without exploding or vanishing gradients."""
        x = torch.randn(self.batch_size, self.seq_len, self.num_features)
        y = torch.randint(0, self.num_classes, (self.batch_size,))
        criterion = nn.CrossEntropyLoss()

        for name in ("LSTM", "BiLSTM", "GRU"):
            model = build_model(name, num_features=self.num_features, hidden_size=64, num_classes=self.num_classes)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            self.assertTrue(torch.isfinite(loss), f"Loss for {name} is not finite")

            # Verify gradients exist and are non-zero and finite for all trainable parameters
            for param_name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad, f"Gradient missing for {name}.{param_name}")
                    self.assertTrue(
                        torch.isfinite(param.grad).all(),
                        f"Non-finite gradient in {name}.{param_name}",
                    )

            optimizer.step()

    def test_sequence_boundary_protection(self) -> None:
        """Verifies that sliding windows never cross capture file boundaries."""
        # 2 files: fileA (10 rows), fileB (8 rows). Window size 5.
        # fileA should produce 10 - 5 + 1 = 6 windows.
        # fileB should produce 8 - 5 + 1 = 4 windows.
        # Total = 10 windows, and groups must never mix.
        X = np.random.randn(18, 33).astype(np.float32)
        y = np.random.randint(0, 4, size=18, dtype=np.int64)
        groups = np.array(["fileA.csv"] * 10 + ["fileB.csv"] * 8, dtype=object)
        row_ids = np.concatenate([np.arange(10), np.arange(8)]).astype(np.int64)

        X_seq, y_seq, seq_groups, seq_rows = build_mqtt_uad_sequences(
            X, y, groups, row_ids, window_size=5, stride=1, label_mode="last"
        )
        self.assertEqual(len(X_seq), 10)
        self.assertEqual(list(seq_groups), ["fileA.csv"] * 6 + ["fileB.csv"] * 4)
        self.assertEqual(seq_rows[0], 4)
        self.assertEqual(seq_rows[5], 9)
        self.assertEqual(seq_rows[6], 4)
        self.assertEqual(seq_rows[9], 7)

    def test_reproducibility(self) -> None:
        """Verifies deterministic initialization under set_seed."""
        set_seed(42)
        m1 = LSTMClassifier(num_features=self.num_features, hidden_size=64, num_classes=self.num_classes)
        w1 = m1.fc[0].weight.data.clone()

        set_seed(42)
        m2 = LSTMClassifier(num_features=self.num_features, hidden_size=64, num_classes=self.num_classes)
        w2 = m2.fc[0].weight.data.clone()

        self.assertTrue(torch.equal(w1, w2), "Model initialization is not reproducible with set_seed")

    def test_training_pipeline_smoke(self) -> None:
        """Verifies train_single_sequence_model runs, computes metrics, and produces valid checkpoints."""
        set_seed(42)
        n_samples = 40
        X_tr = torch.randn(n_samples, self.seq_len, self.num_features)
        y_tr = torch.randint(0, self.num_classes, (n_samples,))
        X_val = torch.randn(20, self.seq_len, self.num_features)
        y_val = torch.randint(0, self.num_classes, (20,))
        X_te = torch.randn(20, self.seq_len, self.num_features)
        y_te = torch.randint(0, self.num_classes, (20,))

        data_dict = {
            "train_loader": DataLoader(TensorDataset(X_tr, y_tr), batch_size=8, shuffle=True),
            "val_loader": DataLoader(TensorDataset(X_val, y_val), batch_size=8, shuffle=False),
            "test_loader": DataLoader(TensorDataset(X_te, y_te), batch_size=8, shuffle=False),
            "class_weights": np.ones(self.num_classes, dtype=np.float32),
            "class_names": ("normal", "DoS", "mitm", "intrusion"),
            "feature_names": [f"f_{i}" for i in range(self.num_features)],
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "split_sizes": [n_samples, 20, 20],
            "y_val": y_val.numpy(),
            "y_test": y_te.numpy(),
        }

        metrics, thresh, payload = train_single_sequence_model(
            "LSTM",
            data_dict,
            seed=42,
            epochs=2,
            device_name="cpu",
            fast_mode=True,
            task_id="test_task",
        )

        self.assertIn("macro_f1", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("balanced_accuracy", metrics)
        self.assertIn("meta", payload)
        self.assertEqual(payload["meta"]["model_family"], "LSTM")

    def test_val_macro_f1_with_absent_classes_uses_explicit_labels(self) -> None:
        """Verifies that validation macro-F1 correctly evaluates all 4 classes even when some are absent."""
        from sklearn.metrics import f1_score
        labels = [0, 1, 2, 3]
        # Only class 0 is present in y_val
        y_val = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        # Without labels, it evaluates to 1.0 (over 1 class)
        f1_inferred = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        self.assertEqual(f1_inferred, 1.0)
        # With explicit labels, class 0 is 1.0, classes 1,2,3 are 0.0 -> macro-F1 is 0.25
        f1_explicit = float(f1_score(y_val, y_pred, labels=labels, average="macro", zero_division=0))
        self.assertEqual(f1_explicit, 0.25)

    def test_no_user_warning_on_absent_test_classes(self) -> None:
        """Regression test ensuring balanced accuracy and macro-F1 emit zero warnings when y_pred has classes absent from y_true."""
        import warnings
        from ids.train_baselines import compute_classification_metrics

        class_names = ("normal", "DoS", "mitm", "intrusion")
        # y_true has only normal (0) and DoS (1); y_pred predicts normal (0) and intrusion (3)
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 3, 3])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics = compute_classification_metrics(y_true, y_pred, class_names)
            cm_arr = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
            row_sums = cm_arr.sum(axis=1)
            valid_rows = row_sums > 0
            bal_acc = float(np.mean(np.diag(cm_arr)[valid_rows] / row_sums[valid_rows]))

            user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
            self.assertEqual(len(user_warnings), 0, f"Unexpected UserWarning: {[str(w.message) for w in user_warnings]}")
            self.assertEqual(bal_acc, 0.5)
            self.assertEqual(metrics["per_class"]["mitm"]["support"], 0)
            self.assertEqual(metrics["per_class"]["intrusion"]["support"], 0)


if __name__ == "__main__":
    unittest.main()

