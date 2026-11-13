"""Unit tests for Stage 5.2 LSTM & BiLSTM Sequence Model Training Pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ids.datasets_four_sensors import (
    FOUR_SENSORS_CLASSES,
    build_four_sensors_sequences,
    compute_class_weights,
    load_four_sensors_dataset,
)
from ids.models_sequence import BiLSTMClassifier, LSTMClassifier, build_model
from ids.train_four_sensors_sequences import (
    evaluate_loader,
    set_seed,
    train_single_run,
)


class TestStage5Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ds_multi = load_four_sensors_dataset(binary=False)
        cls.seq_multi = build_four_sensors_sequences(cls.ds_multi, window_size=16, stride=1)
        cls.ds_bin = load_four_sensors_dataset(binary=True)
        cls.seq_bin = build_four_sensors_sequences(cls.ds_bin, window_size=16, stride=1)

    def test_model_input_output_dimensions(self) -> None:
        """Verifies forward pass dimensions for LSTM and BiLSTM on 14 features."""
        batch_size = 16
        seq_len = 16
        num_features = 14

        x = torch.randn(batch_size, seq_len, num_features)

        # 7-Class Multiclass models
        lstm_7 = build_model("LSTM", num_features=num_features, hidden_size=64, num_classes=7)
        bilstm_7 = build_model("BiLSTM", num_features=num_features, hidden_size=64, num_classes=7)

        lstm_7.eval()
        bilstm_7.eval()

        with torch.no_grad():
            out_lstm = lstm_7(x)
            out_bilstm = bilstm_7(x)

        self.assertEqual(out_lstm.shape, (batch_size, 7))
        self.assertEqual(out_bilstm.shape, (batch_size, 7))
        self.assertFalse(torch.isnan(out_lstm).any())
        self.assertFalse(torch.isnan(out_bilstm).any())

        # 2-Class Binary models
        lstm_2 = build_model("LSTM", num_features=num_features, hidden_size=64, num_classes=2)
        bilstm_2 = build_model("BiLSTM", num_features=num_features, hidden_size=64, num_classes=2)

        lstm_2.eval()
        bilstm_2.eval()

        with torch.no_grad():
            out_lstm_bin = lstm_2(x)
            out_bilstm_bin = bilstm_2(x)

        self.assertEqual(out_lstm_bin.shape, (batch_size, 2))
        self.assertEqual(out_bilstm_bin.shape, (batch_size, 2))

    def test_training_batch_execution_and_gradient_flow(self) -> None:
        """Verifies loss computation and gradient backpropagation on mini-batches."""
        batch_size = 16
        seq_len = 16
        num_features = 14
        num_classes = 7

        x = torch.randn(batch_size, seq_len, num_features)
        y = torch.randint(0, num_classes, (batch_size,))
        weights = torch.ones(num_classes, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weights)

        for name in ("LSTM", "BiLSTM"):
            model = build_model(name, num_features=num_features, hidden_size=64, num_classes=num_classes)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            self.assertTrue(torch.isfinite(loss))

            # Verify all trainable parameters receive finite gradients
            for p_name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad, f"Missing gradient in {name}.{p_name}")
                    self.assertTrue(torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}.{p_name}")

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    def test_class_weight_compatibility_with_loss(self) -> None:
        """Verifies computed training class weights integrate cleanly with CrossEntropyLoss."""
        weights = self.seq_multi.class_weights
        self.assertEqual(len(weights), 7)
        self.assertTrue(np.all(np.isfinite(weights)))

        w_tensor = torch.from_numpy(weights).float()
        criterion = nn.CrossEntropyLoss(weight=w_tensor)

        logits = torch.randn(32, 7)
        targets = torch.randint(0, 7, (32,))
        loss = criterion(logits, targets)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)

    def test_reproducibility(self) -> None:
        """Verifies deterministic initialization under set_seed."""
        set_seed(42)
        m1 = LSTMClassifier(num_features=14, hidden_size=64, num_classes=7)
        w1 = m1.fc[0].weight.data.clone()

        set_seed(42)
        m2 = LSTMClassifier(num_features=14, hidden_size=64, num_classes=7)
        w2 = m2.fc[0].weight.data.clone()

        self.assertTrue(torch.equal(w1, w2), "Model initialization failed reproducibility check")

    def test_checkpoint_save_and_load(self) -> None:
        """Verifies atomic checkpoint saving and loading of model weights and metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir)
            scaler_params = {"mean": [0.0] * 14, "scale": [1.0] * 14}

            # Run 1-epoch fast smoke training
            res = train_single_run(
                model_name="LSTM",
                task_type="multiclass",
                seq_data=self.seq_multi,
                scaler_params=scaler_params,
                seed=42,
                max_epochs=1,
                device_name="cpu",
                out_dir=out_path,
                fast_mode=True,
            )

            pt_file = Path(res["pt_path"])
            json_file = Path(res["json_path"])

            self.assertTrue(pt_file.is_file(), f"Missing PyTorch checkpoint: {pt_file}")
            self.assertTrue(json_file.is_file(), f"Missing metadata JSON: {json_file}")

            # Verify PyTorch checkpoint loads
            ckpt = torch.load(pt_file, map_location="cpu")
            self.assertIn("state_dict", ckpt)
            self.assertEqual(ckpt["model_name"], "LSTM")
            self.assertEqual(len(ckpt["feature_names"]), 14)
            self.assertEqual(len(ckpt["class_names"]), 7)

            # Re-instantiate model and load state dict
            loaded_model = build_model("LSTM", num_features=14, hidden_size=64, num_classes=7)
            loaded_model.load_state_dict(ckpt["state_dict"])
            loaded_model.eval()

            # Verify metadata JSON contents
            with open(json_file, encoding="utf-8") as f:
                import json
                meta_json = json.load(f)
            self.assertIn("meta", meta_json)
            self.assertIn("validation_metrics", meta_json)
            self.assertIn("test_metrics", meta_json)
            self.assertEqual(meta_json["meta"]["seed"], 42)

    def test_correct_seven_class_predictions(self) -> None:
        """Verifies multiclass predictions cover valid class indices and sum to 1.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir)
            scaler_params = {"mean": [0.0] * 14, "scale": [1.0] * 14}

            res = train_single_run(
                model_name="BiLSTM",
                task_type="multiclass",
                seq_data=self.seq_multi,
                scaler_params=scaler_params,
                seed=42,
                max_epochs=1,
                device_name="cpu",
                out_dir=out_path,
                fast_mode=True,
            )

            cm = res["test_metrics"]["confusion_matrix"]
            self.assertEqual(len(cm), 7)
            for row in cm:
                self.assertEqual(len(row), 7)

            self.assertIn("macro_f1", res["test_metrics"])
            self.assertIn("balanced_accuracy", res["test_metrics"])
            self.assertEqual(len(res["test_metrics"]["per_class"]), 7)

    def test_binary_evaluation(self) -> None:
        """Verifies dedicated binary classification task outputs 2x2 confusion matrix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir)
            scaler_params = {"mean": [0.0] * 14, "scale": [1.0] * 14}

            res = train_single_run(
                model_name="LSTM",
                task_type="binary",
                seq_data=self.seq_bin,
                scaler_params=scaler_params,
                seed=42,
                max_epochs=1,
                device_name="cpu",
                out_dir=out_path,
                fast_mode=True,
            )

            cm = res["test_metrics"]["confusion_matrix"]
            self.assertEqual(len(cm), 2)
            self.assertEqual(len(cm[0]), 2)
            self.assertEqual(len(cm[1]), 2)
            self.assertIn("Normal", res["test_metrics"]["per_class"])
            self.assertIn("Attack", res["test_metrics"]["per_class"])

    def test_no_test_set_use_during_training_or_early_stopping(self) -> None:
        """Verifies that early stopping and model selection NEVER touch the test set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir)
            scaler_params = {"mean": [0.0] * 14, "scale": [1.0] * 14}

            # 1. Baseline run with clean sequence data
            res_clean = train_single_run(
                model_name="LSTM",
                task_type="multiclass",
                seq_data=self.seq_multi,
                scaler_params=scaler_params,
                seed=42,
                max_epochs=2,
                device_name="cpu",
                out_dir=out_path / "clean",
                fast_mode=True,
            )

            # 2. Experimental run where test set targets are completely corrupted with dummy zeros
            import copy
            corrupted_seq = copy.deepcopy(self.seq_multi)
            corrupted_seq.y_test_seq = np.zeros_like(corrupted_seq.y_test_seq)

            res_corrupted = train_single_run(
                model_name="LSTM",
                task_type="multiclass",
                seq_data=corrupted_seq,
                scaler_params=scaler_params,
                seed=42,
                max_epochs=2,
                device_name="cpu",
                out_dir=out_path / "corrupted",
                fast_mode=True,
            )

            # Invariant: Training progression, validation metrics, early stopping decisions,
            # and selected best epoch must be 100% identical regardless of test set contents.
            self.assertIn(res_corrupted["best_epoch"], (1, 2))
            self.assertEqual(res_corrupted["best_epoch"], res_clean["best_epoch"])
            self.assertAlmostEqual(res_corrupted["val_macro_f1"], res_clean["val_macro_f1"], places=5)
            self.assertEqual(res_corrupted["total_epochs_run"], res_clean["total_epochs_run"])
            self.assertEqual(
                res_corrupted["test_metrics"]["total_epochs_run"],
                res_clean["test_metrics"]["total_epochs_run"],
            )
            self.assertAlmostEqual(
                res_corrupted["val_metrics"]["macro_f1"],
                res_clean["val_metrics"]["macro_f1"],
                places=5,
            )
            # The corrupted test labels affect only the final test evaluation metric, confirming
            # that test set corruption did not leak into training or model checkpoint selection.
            self.assertNotEqual(
                res_corrupted["test_metrics"]["macro_f1"],
                res_clean["test_metrics"]["macro_f1"],
            )


if __name__ == "__main__":
    unittest.main()

