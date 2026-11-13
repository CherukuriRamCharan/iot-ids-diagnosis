"""Stage 5.2: LSTM and BiLSTM Sequence Model Training & Evaluation Pipeline.

Trains recurrent neural architectures (LSTMClassifier, BiLSTMClassifier) on chronological
four-sensor IoT sequence windows across multiple seeds (42-46) for both 7-class multiclass
and dedicated binary (Normal vs. Attack) classification tasks.

Key Principles:
1. Approved 14 Observable Features: Strictly uses audited physical/network features.
2. Training-Only Class Weights: Inversely weighted cross-entropy loss to counteract acute minority imbalance.
3. Anti-Leakage Early Stopping: Early stopping ($P=5$) and checkpoint selection driven strictly by validation Macro-F1.
4. Independent Test Evaluation: Test set is evaluated strictly ONCE on the selected best checkpoint.
5. Multi-Seed Replication: Evaluates 5 random seeds (42-46), reporting per-seed values, mean, and standard deviation.
6. Safe Checkpointing: Atomic JSON and PyTorch serialization under results/stage_5_hybrid/.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from ids.datasets_four_sensors import (
    FOUR_SENSORS_BINARY_CLASSES,
    FOUR_SENSORS_CLASSES,
    FourSensorsSequences,
    FourSensorsSplitDataset,
    SELECTED_OBSERVABLE_FEATURES,
    build_four_sensors_sequences,
    load_four_sensors_dataset,
    prepare_four_sensors_loaders,
)
from ids.models_sequence import build_model
from ids.train_baselines import compute_classification_metrics
from ids.train_baselines_improved import save_json_atomic

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "results" / "stage_5_hybrid"
DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)


def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility across numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates predicted class indices and softmax probability distributions."""
    model.eval()
    all_preds: List[np.ndarray] = []
    all_probas: List[np.ndarray] = []

    with torch.no_grad():
        for bx, _ in loader:
            bx = bx.to(device)
            logits = model(bx)
            probas = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probas, axis=-1)
            all_preds.append(preds)
            all_probas.append(probas)

    return np.concatenate(all_preds, axis=0), np.concatenate(all_probas, axis=0)


def compute_metrics_with_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """Computes full classification metrics including macro-F1 and balanced accuracy."""
    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    cm_arr = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
    row_sums = cm_arr.sum(axis=1)
    valid_rows = row_sums > 0
    if np.any(valid_rows):
        metrics["balanced_accuracy"] = float(np.mean(np.diag(cm_arr)[valid_rows] / row_sums[valid_rows]))
    else:
        metrics["balanced_accuracy"] = 0.0
    return metrics


def train_single_run(
    model_name: str,
    task_type: str,
    seq_data: FourSensorsSequences,
    scaler_params: Dict[str, Any],
    *,
    seed: int = 42,
    max_epochs: int = 50,
    patience: int = 5,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 128,
    device_name: str = "auto",
    out_dir: Path = DEFAULT_OUT_DIR,
    fast_mode: bool = False,
) -> Dict[str, Any]:
    """Trains a single model instance with validation Macro-F1 early stopping."""
    set_seed(seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    loaders = prepare_four_sensors_loaders(seq_data, batch_size=batch_size)
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]
    y_val = seq_data.y_val_seq
    y_test = seq_data.y_test_seq
    class_names = seq_data.class_names
    num_classes = len(class_names)
    labels = list(range(num_classes))

    model = build_model(
        model_name,
        num_features=len(seq_data.feature_names),
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    # Class-weighted Cross-Entropy loss strictly from training sequence labels
    class_weights_t = torch.from_numpy(seq_data.class_weights).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_val_f1 = -1.0
    best_epoch = 1
    best_weights: Dict[str, Any] = {}
    patience_counter = 0
    actual_epochs = 2 if fast_mode else max_epochs
    actual_patience = 2 if fast_mode else patience

    t0 = time.perf_counter()
    tag = f"{task_type} | {model_name} | seed={seed}"

    for epoch in range(1, actual_epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)

        # Validation evaluation strictly for early stopping
        val_preds, _ = evaluate_loader(model, val_loader, device)
        val_macro_f1 = float(
            f1_score(y_val, val_preds, labels=labels, average="macro", zero_division=0)
        )
        scheduler.step(val_macro_f1)

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= actual_patience:
                break

    train_time_s = time.perf_counter() - t0

    # Load best validation checkpoint
    if best_weights:
        model.load_state_dict(best_weights)
    model.eval()

    # Final Validation evaluation on best weights
    val_preds, val_probas = evaluate_loader(model, val_loader, device)
    val_metrics = compute_metrics_with_balanced_accuracy(y_val, val_preds, class_names)

    # Test evaluation strictly ONCE on the selected best weights
    t_inf_0 = time.perf_counter()
    test_preds, test_probas = evaluate_loader(model, test_loader, device)
    infer_time_s = time.perf_counter() - t_inf_0

    test_metrics = compute_metrics_with_balanced_accuracy(y_test, test_preds, class_names)
    test_metrics["train_time_s"] = round(train_time_s, 3)
    test_metrics["infer_time_s"] = round(infer_time_s, 4)
    test_metrics["best_epoch"] = best_epoch
    test_metrics["total_epochs_run"] = epoch
    test_metrics["best_val_macro_f1"] = round(best_val_f1, 4)
    test_metrics["seed"] = seed
    test_metrics["model"] = model_name
    test_metrics["task"] = task_type

    # Save Checkpoint files under results/stage_5_hybrid/checkpoints/
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{model_name.lower()}_{task_type}_seed{seed}"
    pt_path = ckpt_dir / f"{base_name}.pt"
    json_path = ckpt_dir / f"{base_name}.json"

    # Save PyTorch weights
    torch.save(
        {
            "model_name": model_name,
            "task_type": task_type,
            "seed": seed,
            "state_dict": best_weights,
            "feature_names": list(seq_data.feature_names),
            "class_names": list(class_names),
            "config": {
                "num_features": len(seq_data.feature_names),
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "num_classes": num_classes,
                "dropout": dropout,
            },
        },
        pt_path,
    )

    # Save Checkpoint Metadata and Test Metrics atomically
    checkpoint_meta = {
        "meta": {
            "model_name": model_name,
            "task_type": task_type,
            "seed": seed,
            "best_epoch": best_epoch,
            "fast_mode": fast_mode,
            "feature_names": list(seq_data.feature_names),
            "class_names": list(class_names),
            "class_weights": seq_data.class_weights.tolist(),
            "scaler_params": scaler_params,
            "window_size": seq_data.window_size,
            "stride": seq_data.stride,
            "config": {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "lr": lr,
                "weight_decay": weight_decay,
            },
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    save_json_atomic(json_path, checkpoint_meta)

    # Save Confusion Matrix separately
    cm_dir = out_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    cm_payload = {
        "model": model_name,
        "task": task_type,
        "seed": seed,
        "class_names": list(class_names),
        "confusion_matrix": test_metrics["confusion_matrix"],
    }
    save_json_atomic(cm_dir / f"{base_name}_cm.json", cm_payload)

    return {
        "model": model_name,
        "task": task_type,
        "seed": seed,
        "best_epoch": best_epoch,
        "total_epochs_run": epoch,
        "val_macro_f1": best_val_f1,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "pt_path": str(pt_path),
        "json_path": str(json_path),
    }


def aggregate_runs(
    runs: List[Dict[str, Any]],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    """Computes mean and standard deviation across multi-seed runs."""
    accs = [r["test_metrics"]["accuracy"] for r in runs]
    bal_accs = [r["test_metrics"]["balanced_accuracy"] for r in runs]
    macro_ps = [r["test_metrics"]["macro_precision"] for r in runs]
    macro_rs = [r["test_metrics"]["macro_recall"] for r in runs]
    macro_f1s = [r["test_metrics"]["macro_f1"] for r in runs]
    val_f1s = [r["val_macro_f1"] for r in runs]
    best_epochs = [r["best_epoch"] for r in runs]
    train_times = [r["test_metrics"]["train_time_s"] for r in runs]

    per_class_agg: Dict[str, Dict[str, float]] = {}
    for cname in class_names:
        c_ps = [r["test_metrics"]["per_class"][cname]["precision"] for r in runs]
        c_rs = [r["test_metrics"]["per_class"][cname]["recall"] for r in runs]
        c_f1s = [r["test_metrics"]["per_class"][cname]["f1"] for r in runs]
        c_sup = runs[0]["test_metrics"]["per_class"][cname]["support"]
        per_class_agg[cname] = {
            "precision_mean": round(float(np.mean(c_ps)), 4),
            "precision_std": round(float(np.std(c_ps)), 4),
            "recall_mean": round(float(np.mean(c_rs)), 4),
            "recall_std": round(float(np.std(c_rs)), 4),
            "f1_mean": round(float(np.mean(c_f1s)), 4),
            "f1_std": round(float(np.std(c_f1s)), 4),
            "support": c_sup,
        }

    return {
        "n_seeds": len(runs),
        "seeds": [r["seed"] for r in runs],
        "accuracy_mean": round(float(np.mean(accs)), 4),
        "accuracy_std": round(float(np.std(accs)), 4),
        "balanced_accuracy_mean": round(float(np.mean(bal_accs)), 4),
        "balanced_accuracy_std": round(float(np.std(bal_accs)), 4),
        "macro_precision_mean": round(float(np.mean(macro_ps)), 4),
        "macro_precision_std": round(float(np.std(macro_ps)), 4),
        "macro_recall_mean": round(float(np.mean(macro_rs)), 4),
        "macro_recall_std": round(float(np.std(macro_rs)), 4),
        "macro_f1_mean": round(float(np.mean(macro_f1s)), 4),
        "macro_f1_std": round(float(np.std(macro_f1s)), 4),
        "val_macro_f1_mean": round(float(np.mean(val_f1s)), 4),
        "val_macro_f1_std": round(float(np.std(val_f1s)), 4),
        "best_epoch_mean": round(float(np.mean(best_epochs)), 1),
        "train_time_mean": round(float(np.mean(train_times)), 2),
        "per_class": per_class_agg,
    }


def run_stage5_2_experiments(
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    models: Sequence[str] = ("LSTM", "BiLSTM"),
    out_dir: Path = DEFAULT_OUT_DIR,
    device_name: str = "auto",
    fast_mode: bool = False,
) -> Dict[str, Any]:
    """Orchestrates full Stage 5.2 training across architectures, tasks, and seeds."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_seed_results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        m: {"multiclass": [], "binary": []} for m in models
    }
    aggregated_results: Dict[str, Dict[str, Any]] = {
        m: {} for m in models
    }

    # Load preprocessed datasets
    ds_multi = load_four_sensors_dataset(binary=False)
    seq_multi = build_four_sensors_sequences(ds_multi, window_size=16, stride=1)

    ds_bin = load_four_sensors_dataset(binary=True)
    seq_bin = build_four_sensors_sequences(ds_bin, window_size=16, stride=1)

    scaler_params_multi = {
        "mean": ds_multi.scaler.mean_.tolist() if ds_multi.scaler.mean_ is not None else [],
        "scale": ds_multi.scaler.scale_.tolist() if ds_multi.scaler.scale_ is not None else [],
        "feature_names": list(ds_multi.feature_names),
    }

    scaler_params_bin = {
        "mean": ds_bin.scaler.mean_.tolist() if ds_bin.scaler.mean_ is not None else [],
        "scale": ds_bin.scaler.scale_.tolist() if ds_bin.scaler.scale_ is not None else [],
        "feature_names": list(ds_bin.feature_names),
    }

    print("\n" + "=" * 70)
    print("STAGE 5.2 — LSTM and BiLSTM Multi-Seed Training (Four-Sensor IoT IDS)")
    print(f"Seeds: {list(seeds)} | Models: {list(models)} | Fast Mode: {fast_mode}")
    print("=" * 70)

    comparison_rows = []

    for model_name in models:
        for task_type in ("multiclass", "binary"):
            cur_seq = seq_multi if task_type == "multiclass" else seq_bin
            cur_scaler_params = scaler_params_multi if task_type == "multiclass" else scaler_params_bin
            c_names = FOUR_SENSORS_CLASSES if task_type == "multiclass" else FOUR_SENSORS_BINARY_CLASSES

            runs = []
            for s in seeds:
                res = train_single_run(
                    model_name=model_name,
                    task_type=task_type,
                    seq_data=cur_seq,
                    scaler_params=cur_scaler_params,
                    seed=s,
                    device_name=device_name,
                    out_dir=out_dir,
                    fast_mode=fast_mode,
                )
                runs.append(res)
                all_seed_results[model_name][task_type].append(res)
                print(
                    f"[{model_name} | {task_type} | seed={s}] "
                    f"Best Epoch: {res['best_epoch']} | "
                    f"Val Macro-F1: {res['val_macro_f1']:.4f} | "
                    f"Test Macro-F1: {res['test_metrics']['macro_f1']:.4f} | "
                    f"Test Acc: {res['test_metrics']['accuracy']:.4f}"
                )

            agg = aggregate_runs(runs, c_names)
            aggregated_results[model_name][task_type] = agg

            comparison_rows.append(
                {
                    "model": model_name,
                    "task": task_type,
                    "n_classes": len(c_names),
                    "test_accuracy_mean": agg["accuracy_mean"],
                    "test_accuracy_std": agg["accuracy_std"],
                    "test_balanced_acc_mean": agg["balanced_accuracy_mean"],
                    "test_balanced_acc_std": agg["balanced_accuracy_std"],
                    "test_macro_f1_mean": agg["macro_f1_mean"],
                    "test_macro_f1_std": agg["macro_f1_std"],
                    "val_macro_f1_mean": agg["val_macro_f1_mean"],
                    "val_macro_f1_std": agg["val_macro_f1_std"],
                    "best_epoch_mean": agg["best_epoch_mean"],
                    "train_time_mean_s": agg["train_time_mean"],
                }
            )

    # Save per-seed metrics JSON
    save_json_atomic(out_dir / "per_seed_metrics.json", all_seed_results)

    # Save aggregated metrics JSON
    save_json_atomic(out_dir / "aggregated_metrics.json", aggregated_results)

    # Save model comparison CSV
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(out_dir / "model_comparison.csv", index=False)

    print("\n" + "=" * 70)
    print("STAGE 5.2 TRAINING COMPLETED SUCCESSFULLY")
    print(f"Results saved under: {out_dir}")
    print("=" * 70)

    return {
        "all_seed_results": all_seed_results,
        "aggregated_results": aggregated_results,
        "comparison_df": comp_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5.2 Sequence Model Training")
    parser.add_argument("--fast", action="store_true", help="Run smoke mode with 2 epochs")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cpu', 'cuda', 'auto')")
    args = parser.parse_args()

    run_stage5_2_experiments(fast_mode=args.fast, device_name=args.device)

