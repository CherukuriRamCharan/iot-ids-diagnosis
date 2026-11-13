"""Stage 3C: Deep Learning Sequence Model Training and Evaluation Pipeline.

Implements sequential neural training (LSTM, BiLSTM, GRU) on chronological MQTT-UAD
packet sequences. Reuses Stage 3A leakage-safe sequence generators and masked scalers.
Saves atomic checkpoints and comparative metrics against Stage 3B.6 baselines.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from ids.datasets_real import (
    DEFAULT_UAD_DIR,
    MQTT_UAD_CLASSES,
    UAD_PRESERVE_BINARY_COLS,
    MaskedStandardScaler,
    assert_no_leakage_columns,
    build_mqtt_uad_sequences,
    compute_class_weights,
    load_mqtt_uad,
)
from ids.models_sequence import MODEL_REGISTRY, build_model
from ids.train_baselines import (
    DEFAULT_SEEDS,
    compute_classification_metrics,
)
from ids.train_baselines_improved import (
    default_n_jobs,
    resolve_n_jobs,
    save_json_atomic,
    tune_binary_threshold,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "results" / "stage_3c_sequences"
DEFAULT_REPORT_PATH = ROOT / "docs" / "STAGE3C_SEQUENCE_MODELING_REPORT.md"
STAGE3B6_BASELINE_CSV = ROOT / "results" / "stage_3b6_improvements" / "model_comparison.csv"


@dataclass(frozen=True)
class SequenceTask:
    task_id: str
    label: str
    binary: bool
    mqtt_only: bool


STAGE3C_TASKS: tuple[SequenceTask, ...] = (
    SequenceTask("mqtt_uad_full_multiclass", "MQTT-UAD (Full Stream, 4-Class)", binary=False, mqtt_only=False),
    SequenceTask("mqtt_uad_full_binary", "MQTT-UAD (Full Stream, Binary)", binary=True, mqtt_only=False),
    SequenceTask("mqtt_uad_mqtt_only_multiclass", "MQTT-UAD (MQTT-Only, 4-Class)", binary=False, mqtt_only=True),
    SequenceTask("mqtt_uad_mqtt_only_binary", "MQTT-UAD (MQTT-Only, Binary)", binary=True, mqtt_only=True),
)
STAGE3C_MODELS: tuple[str, ...] = ("LSTM", "BiLSTM", "GRU")


def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility across numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_sequence_loaders(
    uad_dir: Path,
    task: SequenceTask,
    *,
    window_size: int = 16,
    stride: int = 1,
    batch_size: int = 128,
    fast_mode: bool = False,
) -> dict[str, Any]:
    """Loads raw data, fits normalizer on train-only, and builds PyTorch DataLoaders."""
    ds = load_mqtt_uad(uad_dir, mqtt_only=task.mqtt_only, binary=task.binary)
    assert_no_leakage_columns(ds.feature_names)

    # Train-only masked standard scaler
    scaler = MaskedStandardScaler(
        ds.feature_names,
        sentinel_value=-1.0,
        preserve_cols=UAD_PRESERVE_BINARY_COLS,
    )
    X_tr_norm = scaler.fit_transform(ds.X_train)
    X_va_norm = scaler.transform(ds.X_val)
    X_te_norm = scaler.transform(ds.X_test)

    # In fast mode, sample an attack-aware contiguous slice from each capture file to ensure all classes are present
    if fast_mode:
        def _slice_groups(X_arr, y_arr, grp_arr, row_arr, per_group: int):
            keep_idx = []
            for g in np.unique(grp_arr):
                g_mask = grp_arr == g
                g_idx = np.where(g_mask)[0]
                g_y = y_arr[g_idx]
                attack_sub_idx = np.where(g_y > 0)[0]
                if len(attack_sub_idx) > 0:
                    atk_start = attack_sub_idx[0]
                    start = max(0, atk_start - min(50, per_group // 4))
                else:
                    start = 0
                end = min(len(g_idx), start + per_group)
                keep_idx.extend(g_idx[start:end])
            keep_idx = np.sort(keep_idx)
            return X_arr[keep_idx], y_arr[keep_idx], grp_arr[keep_idx], row_arr[keep_idx]

        X_tr_norm, y_tr, grp_tr, row_tr = _slice_groups(X_tr_norm, ds.y_train, ds.train_groups, ds.train_row_ids, 1500)
        X_va_norm, y_va, grp_va, row_va = _slice_groups(X_va_norm, ds.y_val, ds.val_groups, ds.val_row_ids, 500)
        X_te_norm, y_te, grp_te, row_te = _slice_groups(X_te_norm, ds.y_test, ds.test_groups, ds.test_row_ids, 500)
    else:
        y_tr, grp_tr, row_tr = ds.y_train, ds.train_groups, ds.train_row_ids
        y_va, grp_va, row_va = ds.y_val, ds.val_groups, ds.val_row_ids
        y_te, grp_te, row_te = ds.y_test, ds.test_groups, ds.test_row_ids

    # Build sliding windows strictly within each split and file group
    X_tr_s, y_tr_s, _, _ = build_mqtt_uad_sequences(
        X_tr_norm, y_tr, grp_tr, row_tr, window_size=window_size, stride=stride, label_mode="last"
    )
    X_va_s, y_va_s, _, _ = build_mqtt_uad_sequences(
        X_va_norm, y_va, grp_va, row_va, window_size=window_size, stride=stride, label_mode="last"
    )
    X_te_s, y_te_s, _, _ = build_mqtt_uad_sequences(
        X_te_norm, y_te, grp_te, row_te, window_size=window_size, stride=stride, label_mode="last"
    )

    # Class weights from train-only sequence labels
    n_classes = len(ds.class_names)
    weights = compute_class_weights(y_tr_s, n_classes)

    # Torch DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr_s)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va_s), torch.from_numpy(y_va_s)),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te_s), torch.from_numpy(y_te_s)),
        batch_size=batch_size,
        shuffle=False,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_weights": weights,
        "class_names": ds.class_names,
        "feature_names": ds.feature_names,
        "num_features": len(ds.feature_names),
        "num_classes": n_classes,
        "split_sizes": [len(X_tr_s), len(X_va_s), len(X_te_s)],
        "y_val": y_va_s,
        "y_test": y_te_s,
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates predictions and softmax probabilities for a DataLoader."""
    model.eval()
    all_preds: list[np.ndarray] = []
    all_probas: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probas = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probas, axis=-1)
            all_preds.append(preds)
            all_probas.append(probas)
    return np.concatenate(all_preds, axis=0), np.concatenate(all_probas, axis=0)


def train_single_sequence_model(
    model_name: str,
    data_dict: dict[str, Any],
    *,
    seed: int = 42,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device_name: str = "auto",
    fast_mode: bool = False,
    task_id: str = "",
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """Trains a sequence model with validation Macro-F1 early stopping."""
    set_seed(seed)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    model = build_model(
        model_name,
        num_features=data_dict["num_features"],
        hidden_size=64,
        num_layers=2,
        num_classes=data_dict["num_classes"],
        dropout=0.2,
    ).to(device)

    class_w_tensor = torch.from_numpy(data_dict["class_weights"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    train_loader = data_dict["train_loader"]
    val_loader = data_dict["val_loader"]
    y_val = data_dict["y_val"]
    y_test = data_dict["y_test"]
    class_names = data_dict["class_names"]
    labels = list(range(len(class_names)))

    best_val_f1 = -1.0
    best_weights: dict[str, Any] = {}
    patience = 2 if fast_mode else 5
    max_epochs = 2 if fast_mode else epochs
    patience_counter = 0

    t0 = time.perf_counter()
    tag = f"{task_id} | {model_name} | seed={seed}"

    for epoch in range(1, max_epochs + 1):
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

        # Validation evaluation
        val_preds, _ = evaluate_model(model, val_loader, device)
        val_macro_f1 = float(
            f1_score(y_val, val_preds, labels=labels, average="macro", zero_division=0)
        )
        scheduler.step(val_macro_f1)

        print(
            f"  [{tag}] Epoch {epoch}/{max_epochs} - loss: {avg_loss:.4f} - val_macro_f1: {val_macro_f1:.4f}",
            flush=True,
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [{tag}] Early stopping triggered at epoch {epoch}.", flush=True)
                break

    train_time_s = time.perf_counter() - t0

    # Load best weights for test evaluation
    if best_weights:
        model.load_state_dict(best_weights)
    model.eval()

    t_infer_0 = time.perf_counter()
    test_preds, test_probas = evaluate_model(model, data_dict["test_loader"], device)
    infer_time_s = time.perf_counter() - t_infer_0

    metrics = compute_classification_metrics(y_test, test_preds, class_names)
    cm_arr = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
    row_sums = cm_arr.sum(axis=1)
    valid_rows = row_sums > 0
    if np.any(valid_rows):
        metrics["balanced_accuracy"] = float(np.mean(np.diag(cm_arr)[valid_rows] / row_sums[valid_rows]))
    else:
        metrics["balanced_accuracy"] = 0.0
    metrics["train_time_s"] = train_time_s
    metrics["inference_time_s"] = infer_time_s
    metrics["inference_ms_per_1k"] = (1000.0 * infer_time_s / max(len(y_test), 1)) * 1000.0
    metrics["best_val_macro_f1"] = best_val_f1
    metrics["selected_config"] = {
        "model_name": model_name,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "lr": lr,
    }
    metrics["task_id"] = task_id
    metrics["model"] = model_name
    metrics["seed"] = seed

    # Binary threshold tuning on validation set only
    thresh_info: dict[str, Any] | None = None
    if len(class_names) == 2:
        _, val_probas = evaluate_model(model, val_loader, device)
        val_proba_pos = val_probas[:, 1]
        tau, val_tau_f1 = tune_binary_threshold(y_val, val_proba_pos)
        test_proba_pos = test_probas[:, 1]
        pred_tau = (test_proba_pos >= tau).astype(np.int64)

        m_default = compute_classification_metrics(y_test, test_preds, class_names)
        m_tuned = compute_classification_metrics(y_test, pred_tau, class_names)

        thresh_info = {
            "selected_threshold": tau,
            "val_macro_f1_at_tau": val_tau_f1,
            "default_0.5_test_macro_f1": m_default["macro_f1"],
            "default_0.5_attack_precision": m_default["per_class"]["attack"]["precision"],
            "default_0.5_attack_recall": m_default["per_class"]["attack"]["recall"],
            "default_0.5_attack_f1": m_default["per_class"]["attack"]["f1"],
            "default_0.5_confusion_matrix": m_default["confusion_matrix"],
            "tuned_test_macro_f1": m_tuned["macro_f1"],
            "tuned_attack_precision": m_tuned["per_class"]["attack"]["precision"],
            "tuned_attack_recall": m_tuned["per_class"]["attack"]["recall"],
            "tuned_attack_f1": m_tuned["per_class"]["attack"]["f1"],
            "tuned_confusion_matrix": m_tuned["confusion_matrix"],
        }

    checkpoint_payload = {
        "meta": {
            "task_id": task_id,
            "model_family": model_name,
            "seed": int(seed),
            "fast_mode": bool(fast_mode),
            "feature_names": list(data_dict["feature_names"]),
            "class_names": list(class_names),
            "split_sizes": data_dict["split_sizes"],
        },
        "metrics": metrics,
        "threshold_info": thresh_info,
    }

    return metrics, thresh_info, checkpoint_payload


def run_stage3c(
    *,
    uad_dir: Path = DEFAULT_UAD_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    baseline_csv: Path = STAGE3B6_BASELINE_CSV,
    tasks: Sequence[SequenceTask] = STAGE3C_TASKS,
    models: Sequence[str] = STAGE3C_MODELS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    epochs: int = 20,
    batch_size: int = 128,
    window_size: int = 16,
    stride: int = 1,
    fast_mode: bool = False,
    resume: bool = False,
    aggregate_only: bool = False,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Runs complete Stage 3C sequence experiments with atomic checkpoints."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    if not aggregate_only:
        for task in tasks:
            print(f"\n[STAGE 3C] Preparing sequences for {task.task_id}...", flush=True)
            data_dict = prepare_sequence_loaders(
                uad_dir,
                task,
                window_size=window_size,
                stride=stride,
                batch_size=batch_size,
                fast_mode=fast_mode,
            )

            for m_name in models:
                for seed in seeds:
                    ckpt_file = ckpt_dir / f"{task.task_id}__{m_name}__seed{seed}.json"
                    if resume and ckpt_file.exists():
                        try:
                            d = json.loads(ckpt_file.read_text(encoding="utf-8"))
                            if d.get("meta", {}).get("fast_mode") == bool(fast_mode):
                                print(f"[RESUME] Skipping completed checkpoint {ckpt_file.name}", flush=True)
                                continue
                        except Exception:
                            pass

                    print(f"\n[TRAIN] {task.task_id} | {m_name} | seed={seed}", flush=True)
                    m, th, payload = train_single_sequence_model(
                        m_name,
                        data_dict,
                        seed=seed,
                        epochs=epochs,
                        device_name=device_name,
                        fast_mode=fast_mode,
                        task_id=task.task_id,
                    )
                    save_json_atomic(ckpt_file, payload)

    return {"status": "complete", "out_dir": str(out_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3C sequence modeling experiments.")
    parser.add_argument("--uad-dir", type=Path, default=DEFAULT_UAD_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated task IDs.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names (LSTM,BiLSTM,GRU).")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fast", action="store_true", help="Run 2-epoch smoke test on subset.")
    parser.add_argument("--resume", action="store_true", help="Skip completed checkpoints.")
    parser.add_argument("--aggregate-only", action="store_true", help="Only generate report from checkpoints.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda.")
    args = parser.parse_args()

    task_map = {t.task_id: t for t in STAGE3C_TASKS}
    sel_tasks = [task_map[t.strip()] for t in args.tasks.split(",")] if args.tasks else STAGE3C_TASKS
    sel_models = [m.strip() for m in args.models.split(",")] if args.models else STAGE3C_MODELS

    run_stage3c(
        uad_dir=args.uad_dir,
        out_dir=args.out_dir,
        report_path=args.report_path,
        tasks=sel_tasks,
        models=sel_models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fast_mode=args.fast,
        resume=args.resume,
        aggregate_only=args.aggregate_only,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()

