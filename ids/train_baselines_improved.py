"""Stage 3B.6: Controlled baseline improvement and accuracy optimization on MQTT-UAD.

Preserves Stage 3B results (results/stage_3b_baselines/) as an immutable baseline.
Uses Stage 3A leakage-safe loaders (ids/datasets_real.py) and performs:
1. Validation-only model selection across RandomForest, ExtraTrees,
   HistGradientBoosting, and training-only capped random oversampling.
2. Validation-only binary decision threshold tuning.
3. Scientifically justified feature ablation on MQTT-UAD Full-Stream 4-Class.
4. 5-seed reproducibility (seeds 42..46) with Macro-F1, Balanced Accuracy,
   Accuracy, Per-Class P/R/F1/Support, Confusion Matrices, and Timing.
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
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import balanced_accuracy_score, f1_score

from ids.datasets_real import (
    DEFAULT_UAD_DIR,
    SplitDataset,
    assert_no_leakage_columns,
    load_mqtt_uad,
)
from ids.train_baselines import (
    DEFAULT_SEEDS,
    aggregate_seed_runs,
    compute_classification_metrics,
    save_confusion_matrix_plot,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE_CSV = ROOT / "results" / "stage_3b_baselines" / "model_comparison.csv"
DEFAULT_OUT_DIR = ROOT / "results" / "stage_3b6_improvements"
DEFAULT_REPORT_PATH = ROOT / "docs" / "STAGE3B6_BASELINE_IMPROVEMENT_REPORT.md"


@dataclass(frozen=True)
class UADTask:
    task_id: str
    label: str
    binary: bool
    mqtt_only: bool


UAD_TASKS: tuple[UADTask, ...] = (
    UADTask("mqtt_uad_full_multiclass", "MQTT-UAD (Full Stream, 4-Class)", binary=False, mqtt_only=False),
    UADTask("mqtt_uad_full_binary", "MQTT-UAD (Full Stream, Binary)", binary=True, mqtt_only=False),
    UADTask("mqtt_uad_mqtt_only_multiclass", "MQTT-UAD (MQTT-Only, 4-Class)", binary=False, mqtt_only=True),
    UADTask("mqtt_uad_mqtt_only_binary", "MQTT-UAD (MQTT-Only, Binary)", binary=True, mqtt_only=True),
)
ALL_FAMILIES: tuple[str, ...] = ("RF_tuned", "ExtraTrees_tuned", "HistGB_tuned", "RF_train_oversampled")


def default_n_jobs() -> int:
    """Returns bounded CPU worker count (cpu_count - 2, minimum 1) to avoid oversubscription."""
    return max(1, (os.cpu_count() or 4) - 2)


def resolve_n_jobs(n_jobs: int | None) -> int:
    """Normalizes user-provided n_jobs, preventing -1 oversubscription by default."""
    if n_jobs is None:
        return default_n_jobs()
    if n_jobs == -1:
        return default_n_jobs()
    return max(1, int(n_jobs))


def save_json_atomic(path: Path, payload: Any) -> None:
    """Writes JSON to a temporary file and atomically replaces the target path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def derive_binary_dataset(ds_mc: SplitDataset) -> SplitDataset:
    """Derives a binary (normal vs attack) SplitDataset in memory from a multiclass SplitDataset."""
    return SplitDataset(
        name=f"{ds_mc.name}_binary",
        X_train=ds_mc.X_train,
        y_train=(ds_mc.y_train != 0).astype(np.int64),
        X_val=ds_mc.X_val,
        y_val=(ds_mc.y_val != 0).astype(np.int64),
        X_test=ds_mc.X_test,
        y_test=(ds_mc.y_test != 0).astype(np.int64),
        feature_names=ds_mc.feature_names,
        class_names=("normal", "attack"),
        train_groups=ds_mc.train_groups,
        val_groups=ds_mc.val_groups,
        test_groups=ds_mc.test_groups,
        train_row_ids=ds_mc.train_row_ids,
        val_row_ids=ds_mc.val_row_ids,
        test_row_ids=ds_mc.test_row_ids,
        metadata=ds_mc.metadata,
    )


def build_checkpoint_meta(
    ds: SplitDataset,
    task_id: str,
    model_family: str,
    seed: int,
    fast_mode: bool,
) -> dict[str, Any]:
    """Builds strict compatibility metadata stored inside each checkpoint."""
    return {
        "task_id": task_id,
        "model_family": model_family,
        "seed": int(seed),
        "fast_mode": bool(fast_mode),
        "dataset_name": ds.name,
        "feature_names": list(ds.feature_names),
        "class_names": list(ds.class_names),
        "split_sizes": [int(len(ds.X_train)), int(len(ds.X_val)), int(len(ds.X_test))],
    }


def load_valid_checkpoint(path: Path, expected_meta: dict[str, Any]) -> dict[str, Any] | None:
    """Loads a checkpoint only if valid JSON, complete, and metadata matches exactly."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Checkpoint root is not a dict")
        meta = data.get("meta")
        metrics = data.get("metrics")
        if not isinstance(meta, dict) or not isinstance(metrics, dict):
            raise ValueError("Missing meta or metrics")
        if "selected_config" not in metrics or "macro_f1" not in metrics:
            raise ValueError("Incomplete metrics payload")
        for k, v in expected_meta.items():
            if meta.get(k) != v:
                raise ValueError(f"Metadata mismatch for '{k}': {meta.get(k)} != {v}")
        return data
    except Exception as exc:
        print(f"[WARN] Rejecting incompatible/corrupted checkpoint {path.name}: {exc}", flush=True)
        return None


def cap_oversample_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_cap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Training-only random oversampling of minority classes up to target_cap."""
    rng = np.random.default_rng(seed)
    indices: list[np.ndarray] = []
    for cls in np.unique(y_train):
        cls_idx = np.where(y_train == cls)[0]
        if len(cls_idx) < target_cap:
            extra = rng.choice(cls_idx, size=target_cap - len(cls_idx), replace=True)
            indices.append(np.concatenate([cls_idx, extra]))
        else:
            indices.append(cls_idx)
    all_idx = np.concatenate(indices)
    rng.shuffle(all_idx)
    return X_train[all_idx], y_train[all_idx]


def tune_binary_threshold(
    y_val: np.ndarray,
    val_proba_pos: np.ndarray,
    grid: Sequence[float] = tuple(np.round(np.arange(0.10, 0.91, 0.05), 2)),
) -> tuple[float, float]:
    """Selects binary decision threshold maximizing validation macro-F1."""
    best_tau = 0.50
    best_f1 = -1.0
    for tau in grid:
        pred = (val_proba_pos >= tau).astype(np.int64)
        f1 = float(f1_score(y_val, pred, labels=[0, 1], average="macro", zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
    return best_tau, best_f1


def _eval_predictions(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    train_time_s: float,
    infer_time_s: float,
    n_test: int,
) -> dict[str, Any]:
    m = compute_classification_metrics(y_test, y_pred, class_names)
    m["balanced_accuracy"] = float(balanced_accuracy_score(y_test, y_pred))
    m["train_time_s"] = train_time_s
    m["inference_time_s"] = infer_time_s
    m["inference_ms_per_1k"] = (1000.0 * infer_time_s / max(n_test, 1)) * 1000.0
    return m


def fit_select_and_eval(
    ds: SplitDataset,
    model_family: str,
    seed: int,
    *,
    binary: bool = False,
    n_jobs: int | None = None,
    fast_mode: bool = False,
    task_id: str = "",
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Tunes hyperparameters on validation macro-F1 (train-only fit) and evaluates on test."""
    assert_no_leakage_columns(ds.feature_names)
    workers = resolve_n_jobs(n_jobs)
    labels = list(range(len(ds.class_names)))
    tag = f"{task_id} | {model_family} | seed={seed}" if task_id else f"{model_family} | seed={seed}"

    best_val_f1 = -1.0
    best_cfg: dict[str, Any] = {}
    best_clf: Any = None
    t0 = time.perf_counter()

    def _run_grid(grid: list[dict[str, Any]], build_clf, X_tr: np.ndarray, y_tr: np.ndarray) -> None:
        nonlocal best_val_f1, best_cfg, best_clf
        total = len(grid)
        for idx, cfg in enumerate(grid, start=1):
            t_fit = time.perf_counter()
            clf = build_clf(cfg)
            clf.fit(X_tr, y_tr)
            val_f1 = float(f1_score(ds.y_val, clf.predict(ds.X_val), labels=labels, average="macro", zero_division=0))
            dt = time.perf_counter() - t_fit
            print(f"  [{tag}] fit {idx}/{total} cfg={cfg} -> val_macro_f1={val_f1:.4f} ({dt:.2f}s)", flush=True)
            if val_f1 > best_val_f1:
                best_val_f1, best_cfg, best_clf = val_f1, cfg, clf

    if model_family == "RF_tuned":
        n_est = 40 if fast_mode else 150
        grid = (
            [{"class_weight": None, "max_depth": 20, "min_samples_leaf": 2}]
            if fast_mode
            else [
                {"class_weight": cw, "max_depth": md, "min_samples_leaf": msl}
                for cw in (None, "balanced", "balanced_subsample")
                for md in (16, 24, None)
                for msl in (1, 3)
            ]
        )
        _run_grid(
            grid,
            lambda cfg: RandomForestClassifier(
                n_estimators=n_est,
                max_depth=cfg["max_depth"],
                min_samples_leaf=cfg["min_samples_leaf"],
                class_weight=cfg["class_weight"],
                random_state=seed,
                n_jobs=workers,
            ),
            ds.X_train,
            ds.y_train,
        )

    elif model_family == "ExtraTrees_tuned":
        n_est = 40 if fast_mode else 150
        grid = (
            [{"class_weight": None, "max_depth": 20, "min_samples_leaf": 2}]
            if fast_mode
            else [
                {"class_weight": cw, "max_depth": md, "min_samples_leaf": msl}
                for cw in (None, "balanced", "balanced_subsample")
                for md in (20, None)
                for msl in (1, 3)
            ]
        )
        _run_grid(
            grid,
            lambda cfg: ExtraTreesClassifier(
                n_estimators=n_est,
                max_depth=cfg["max_depth"],
                min_samples_leaf=cfg["min_samples_leaf"],
                class_weight=cfg["class_weight"],
                random_state=seed,
                n_jobs=workers,
            ),
            ds.X_train,
            ds.y_train,
        )

    elif model_family == "HistGB_tuned":
        grid = (
            [{"class_weight": None, "max_depth": 6, "learning_rate": 0.1}]
            if fast_mode
            else [
                {"class_weight": cw, "max_depth": md, "learning_rate": lr}
                for cw in (None, "balanced")
                for md in (6, 10, None)
                for lr in (0.05, 0.1)
            ]
        )
        _run_grid(
            grid,
            lambda cfg: HistGradientBoostingClassifier(
                max_depth=cfg["max_depth"],
                learning_rate=cfg["learning_rate"],
                max_iter=60 if fast_mode else 150,
                class_weight=cfg["class_weight"],
                random_state=seed,
            ),
            ds.X_train,
            ds.y_train,
        )

    elif model_family == "RF_train_oversampled":
        cap = 20 if fast_mode else 10_000
        n_est = 40 if fast_mode else 150
        X_os, y_os = cap_oversample_train(ds.X_train, ds.y_train, target_cap=cap, seed=seed)
        grid = (
            [{"max_depth": 20, "min_samples_leaf": 2, "oversample_cap": cap}]
            if fast_mode
            else [{"max_depth": md, "min_samples_leaf": msl, "oversample_cap": cap} for md in (16, 24, None) for msl in (1, 3)]
        )
        _run_grid(
            grid,
            lambda cfg: RandomForestClassifier(
                n_estimators=n_est,
                max_depth=cfg["max_depth"],
                min_samples_leaf=cfg["min_samples_leaf"],
                class_weight=None,
                random_state=seed,
                n_jobs=workers,
            ),
            X_os,
            y_os,
        )
    else:
        raise ValueError(f"Unknown model_family: {model_family}")

    train_time_s = time.perf_counter() - t0
    assert best_clf is not None

    t1 = time.perf_counter()
    test_pred = best_clf.predict(ds.X_test)
    infer_time_s = time.perf_counter() - t1

    metrics = _eval_predictions(
        ds.y_test, test_pred, ds.class_names, train_time_s, infer_time_s, len(ds.X_test)
    )
    metrics["best_val_macro_f1"] = best_val_f1
    metrics["selected_config"] = {k: (str(v) if v is None else v) for k, v in best_cfg.items()}

    thresh_info: dict[str, Any] | None = None
    if binary and hasattr(best_clf, "predict_proba"):
        val_proba = best_clf.predict_proba(ds.X_val)[:, 1]
        tau, val_tau_f1 = tune_binary_threshold(ds.y_val, val_proba)
        test_proba = best_clf.predict_proba(ds.X_test)[:, 1]
        pred_tau = (test_proba >= tau).astype(np.int64)
        m_default = compute_classification_metrics(ds.y_test, test_pred, ds.class_names)
        m_tuned = compute_classification_metrics(ds.y_test, pred_tau, ds.class_names)
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

    return metrics, thresh_info


def run_feature_ablation(
    ds_full: SplitDataset,
    *,
    seed: int = 42,
    n_jobs: int = -1,
    fast_mode: bool = False,
) -> list[dict[str, Any]]:
    """Evaluates 5 scientifically justified feature subsets on MQTT-UAD Full-Stream 4-Class."""
    fnames = list(ds_full.feature_names)
    layer_flags = {"has_ip", "has_tcp", "is_mqtt_port"}
    mitm_nan_cols = {"mqtt.msgid", "mqtt.sub.qos", "mqtt.suback.qos"}
    packet_cols = {"frame.time_delta", "frame.time_delta_displayed", "frame.cap_len", "frame.len"} | layer_flags

    subsets = {
        "all_33_features": fnames,
        "no_layer_flags": [c for c in fnames if c not in layer_flags],
        "no_mitm_nan_cols": [c for c in fnames if c not in mitm_nan_cols],
        "packet_and_layer_only": [c for c in fnames if c in packet_cols],
        "mqtt_features_only": [c for c in fnames if c not in packet_cols],
    }

    results: list[dict[str, Any]] = []
    for sub_name, cols in subsets.items():
        if not cols:
            continue
        idx = [fnames.index(c) for c in cols]
        sub_ds = SplitDataset(
            name=f"{ds_full.name}_{sub_name}",
            X_train=ds_full.X_train[:, idx],
            y_train=ds_full.y_train,
            X_val=ds_full.X_val[:, idx],
            y_val=ds_full.y_val,
            X_test=ds_full.X_test[:, idx],
            y_test=ds_full.y_test,
            feature_names=tuple(cols),
            class_names=ds_full.class_names,
            train_groups=ds_full.train_groups,
            val_groups=ds_full.val_groups,
            test_groups=ds_full.test_groups,
            train_row_ids=ds_full.train_row_ids,
            val_row_ids=ds_full.val_row_ids,
            test_row_ids=ds_full.test_row_ids,
        )
        clf = HistGradientBoostingClassifier(
            max_depth=10,
            learning_rate=0.1,
            max_iter=60 if fast_mode else 150,
            random_state=seed,
        )
        t0 = time.perf_counter()
        clf.fit(sub_ds.X_train, sub_ds.y_train)
        tr_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        pred = clf.predict(sub_ds.X_test)
        inf_s = time.perf_counter() - t1
        m = _eval_predictions(sub_ds.y_test, pred, sub_ds.class_names, tr_s, inf_s, len(sub_ds.X_test))
        results.append(
            {
                "ablation_id": sub_name,
                "n_features": len(cols),
                "macro_f1": m["macro_f1"],
                "balanced_accuracy": m["balanced_accuracy"],
                "accuracy": m["accuracy"],
                "normal_f1": m["per_class"]["normal"]["f1"],
                "DoS_f1": m["per_class"]["DoS"]["f1"],
                "mitm_f1": m["per_class"]["mitm"]["f1"],
                "intrusion_f1": m["per_class"]["intrusion"]["f1"],
                "mitm_recall": m["per_class"]["mitm"]["recall"],
                "intrusion_recall": m["per_class"]["intrusion"]["recall"],
            }
        )
    return results


def _aggregate_with_bal_acc(seed_runs: list[dict[str, Any]], class_names: Sequence[str]) -> dict[str, Any]:
    agg = aggregate_seed_runs(seed_runs, class_names)
    ba = np.array([r["balanced_accuracy"] for r in seed_runs], dtype=np.float64)
    agg["balanced_accuracy_mean"] = float(np.mean(ba))
    agg["balanced_accuracy_std"] = float(np.std(ba, ddof=0))
    return agg


def generate_improvement_report(
    aggregated: dict[str, Any],
    ablation_rows: list[dict[str, Any]],
    threshold_summary: dict[str, Any],
    baseline_csv: Path,
    seeds: Sequence[int],
    report_path: Path,
    incomplete_experiments: Sequence[str] = (),
) -> None:
    """Writes docs/STAGE3B6_BASELINE_IMPROVEMENT_REPORT.md from verified computed outputs."""
    base_df = pd.read_csv(baseline_csv) if baseline_csv.exists() else pd.DataFrame()
    base_agg_path = baseline_csv.parent / "aggregated_metrics.json"
    base_agg = json.loads(base_agg_path.read_text(encoding="utf-8")) if base_agg_path.exists() else {}
    lines: list[str] = [
        "# Stage 3B.6 — Controlled Baseline Improvement Report (MQTT-UAD)",
        "",
        f"> **Seeds:** `{list(seeds)}` | **Primary Selection Metric:** Validation `Macro-F1`",
        "> **Immutable Reference:** `results/stage_3b_baselines/` (unmodified). All evaluations remain within-capture chronological holdout on MQTT-UAD.",
        "",
    ]
    if incomplete_experiments:
        lines.extend([
            "> **WARNING — INCOMPLETE EXPERIMENT RUN:** The following `(task, model)` pairs are missing one or more seeds and are excluded from aggregated tables until all seeds complete:",
            *[f"> - `{item}`" for item in incomplete_experiments],
            "",
        ])
    lines.extend([
        "## 1. Original Stage 3B vs. Improved Stage 3B.6 Comparison",
        "",
        "| Task | Model | Macro-F1 (mean ± std) | Balanced Acc | Accuracy | `mitm` F1 | `intrusion` F1 | Train (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])

    if not base_df.empty:
        if "dataset_group" in base_df.columns:
            uad_base = base_df[base_df["dataset_group"] == "MQTT-UAD"]
        elif "task_id" in base_df.columns:
            uad_base = base_df[base_df["task_id"].astype(str).str.startswith("mqtt_uad")]
        elif "task" in base_df.columns:
            uad_base = base_df[base_df["task"].astype(str).str.startswith("mqtt_uad")]
        elif "dataset_label" in base_df.columns:
            uad_base = base_df[base_df["dataset_label"].astype(str).str.startswith("MQTT-UAD")]
        else:
            uad_base = pd.DataFrame()
            lines.append(
                f"| N/A | [Stage 3B] N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
            )
            lines.append("")
            lines.append(
                "> **Note:** Baseline comparison marked `N/A` because `dataset_group` could not be determined from available baseline CSV columns."
            )

        for _, r in uad_base.iterrows():
            task_id = r["task_id"] if "task_id" in r and pd.notna(r["task_id"]) else (
                r["task"] if "task" in r and pd.notna(r["task"]) else None
            )
            dataset_label = (
                str(r["dataset_label"])
                if "dataset_label" in r and pd.notna(r["dataset_label"])
                else (str(task_id) if task_id is not None else "N/A")
            )
            model_name = str(r["model"]) if "model" in r and pd.notna(r["model"]) else "N/A"
            bkey = f"{task_id}__{model_name}" if task_id is not None else ""
            b_entry = base_agg.get(bkey, {}) if bkey else {}
            b_pc = b_entry.get("per_class", {})
            if b_entry:
                b_mitm = f"{b_pc['mitm']['f1_mean']:.4f}" if "mitm" in b_pc else "—"
                b_intr = f"{b_pc['intrusion']['f1_mean']:.4f}" if "intrusion" in b_pc else "—"
            else:
                b_mitm = "N/A"
                b_intr = "N/A"
            b_bal = "N/A"
            if "confusion_matrix_sum" in b_entry:
                cm_b = np.asarray(b_entry["confusion_matrix_sum"], dtype=np.float64)
                row_sums = cm_b.sum(axis=1)
                valid_r = row_sums > 0
                if np.any(valid_r):
                    b_bal = f"{float(np.mean(np.diag(cm_b)[valid_r] / row_sums[valid_r])):.4f}"
            elif "balanced_accuracy_mean" in r and pd.notna(r["balanced_accuracy_mean"]):
                b_bal = f"{float(r['balanced_accuracy_mean']):.4f}"

            if "macro_f1_mean" in r and pd.notna(r["macro_f1_mean"]):
                if "macro_f1_std" in r and pd.notna(r["macro_f1_std"]):
                    b_macro_f1 = f"{float(r['macro_f1_mean']):.4f} ± {float(r['macro_f1_std']):.4f}"
                else:
                    b_macro_f1 = f"{float(r['macro_f1_mean']):.4f} ± N/A"
            else:
                b_macro_f1 = "N/A"

            b_acc = (
                f"{float(r['accuracy_mean']):.4f}"
                if "accuracy_mean" in r and pd.notna(r["accuracy_mean"])
                else "N/A"
            )
            b_train_s = (
                f"{float(r['train_time_s_mean']):.2f}"
                if "train_time_s_mean" in r and pd.notna(r["train_time_s_mean"])
                else "N/A"
            )
            lines.append(
                f"| {dataset_label} | [Stage 3B] {model_name} | "
                f"{b_macro_f1} | {b_bal} | {b_acc} | {b_mitm} | {b_intr} | {b_train_s} |"
            )

    for key, agg in aggregated.items():
        pc = agg["per_class"]
        mitm_f1 = f"{pc['mitm']['f1_mean']:.4f}" if "mitm" in pc else "—"
        intr_f1 = f"{pc['intrusion']['f1_mean']:.4f}" if "intrusion" in pc else "—"
        lines.append(
            f"| {agg['task_label']} | [Stage 3B.6] {agg['model']} | "
            f"{agg['macro_f1_mean']:.4f} ± {agg['macro_f1_std']:.4f} | "
            f"{agg['balanced_accuracy_mean']:.4f} ± {agg['balanced_accuracy_std']:.4f} | "
            f"{agg['accuracy_mean']:.4f} | {mitm_f1} | {intr_f1} | {agg['train_time_s_mean']:.2f} |"
        )

    has_sleep_anomaly = any(agg.get("train_time_s_mean", 0) > 3600 for agg in aggregated.values())
    if has_sleep_anomaly:
        lines.append("")
        lines.append(
            "> **Note on Reported Training Times:** For `[Stage 3B.6] RF_tuned` on Full Stream Binary, Seed 44 ran overnight and recorded a wall-clock elapsed time of 41,561.36s due to system suspension; the remaining 4 seeds recorded elapsed wall-clock times averaging 456.12s (Seed 42: 429.85s, Seed 43: 426.35s, Seed 45: 482.88s, Seed 46: 485.39s). Actual CPU process time was not recorded."
        )

    lines.extend([
        "",
        "## 2. Feature Ablation on MQTT-UAD Full-Stream 4-Class (`HistGradientBoosting`, Seed 42)",
        "",
        "| Subset | # Features | Macro-F1 | Balanced Acc | `DoS` F1 | `mitm` F1 (Recall) | `intrusion` F1 (Recall) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for r in ablation_rows:
        lines.append(
            f"| `{r['ablation_id']}` | {r['n_features']} | {r['macro_f1']:.4f} | {r['balanced_accuracy']:.4f} | "
            f"{r['DoS_f1']:.4f} | {r['mitm_f1']:.4f} ({r['mitm_recall']:.4f}) | {r['intrusion_f1']:.4f} ({r['intrusion_recall']:.4f}) |"
        )

    lines.extend([
        "",
        "## 3. Validation-Only Binary Threshold Tuning Summary",
        "",
        "| Task | Model | Selected `tau*` (mean) | Default `0.50` Macro-F1 | Tuned `tau*` Macro-F1 | Attack P / R (`0.50` -> `tau*`) |",
        "|---|---|---:|---:|---:|---|",
    ])
    for k, t in threshold_summary.items():
        lines.append(
            f"| {t['task_label']} | {t['model']} | {t['tau_mean']:.2f} | "
            f"{t['default_f1_mean']:.4f} | {t['tuned_f1_mean']:.4f} | "
            f"{t['default_p_mean']:.4f}/{t['default_r_mean']:.4f} -> {t['tuned_p_mean']:.4f}/{t['tuned_r_mean']:.4f} |"
        )

    lines.extend([
        "",
        "## 4. Per-Class Test Metrics (Stage 3B.6 Models)",
        "",
    ])
    for key, agg in aggregated.items():
        lines.append(f"### {agg['task_label']} — {agg['model']}")
        lines.append("| Class | Test Support | Precision (mean ± std) | Recall (mean ± std) | F1 (mean ± std) |")
        lines.append("|---|---:|---:|---:|---:|")
        for cname, cstats in agg["per_class"].items():
            lines.append(
                f"| `{cname}` | {cstats['support']:,} | "
                f"{cstats['precision_mean']:.4f} ± {cstats['precision_std']:.4f} | "
                f"{cstats['recall_mean']:.4f} ± {cstats['recall_std']:.4f} | "
                f"{cstats['f1_mean']:.4f} ± {cstats['f1_std']:.4f} |"
            )
        lines.append("")

    lines.extend([
        "## 5. Limitations & Scientific Scope",
        "1. **Within-capture chronological evaluation only:** Does not establish unseen-capture or unseen-broker generalization.",
        "2. **MQTT-Only `mitm` test support (`13` samples):** Too small for statistically reliable `mitm` evaluation in `MQTT-Only` mode.",
        "3. **No formal fault-localization labels:** Classification improvements do not imply TFSM fault localization.",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _get_task_dataset(
    task: UADTask,
    uad_dir: Path,
    cache: dict[bool, SplitDataset],
) -> SplitDataset:
    """Loads each raw MQTT-UAD subset once and derives binary splits in memory."""
    if task.mqtt_only not in cache:
        cache[task.mqtt_only] = load_mqtt_uad(uad_dir, mqtt_only=task.mqtt_only, binary=False)
    ds_mc = cache[task.mqtt_only]
    return derive_binary_dataset(ds_mc) if task.binary else ds_mc


def run_stage3b6(
    *,
    uad_dir: Path = DEFAULT_UAD_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    baseline_csv: Path = BASELINE_CSV,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    tasks: Sequence[UADTask] = UAD_TASKS,
    families: Sequence[str] = ALL_FAMILIES,
    n_jobs: int | None = None,
    fast_mode: bool = False,
    resume: bool = False,
    aggregate_only: bool = False,
    expected_tasks: Sequence[UADTask] | None = None,
    expected_families: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Runs Stage 3B.6 controlled improvement experiments with atomic checkpointing."""
    workers = resolve_n_jobs(n_jobs)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cm_dir = out_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    full_tasks = tuple(expected_tasks) if expected_tasks is not None else tuple(tasks)
    full_families = tuple(expected_families) if expected_families is not None else tuple(families)
    ds_cache: dict[bool, SplitDataset] = {}

    # 1. Execute requested chunk (unless aggregate_only=True)
    if not aggregate_only:
        total_runs = len(tasks) * len(families) * len(seeds)
        done_runs = 0
        for task in tasks:
            ds = _get_task_dataset(task, uad_dir, ds_cache)
            if task.task_id == "mqtt_uad_full_multiclass":
                abl_path = ckpt_dir / f"feature_ablation__seed{seeds[0]}.json"
                abl_loaded = None
                if resume and abl_path.exists():
                    try:
                        cand = json.loads(abl_path.read_text(encoding="utf-8"))
                        if isinstance(cand, dict) and cand.get("fast_mode") == bool(fast_mode) and cand.get("seed") == int(seeds[0]):
                            abl_loaded = cand["rows"]
                            print(f"[RESUME] Reusing feature ablation checkpoint ({abl_path.name})", flush=True)
                    except Exception:
                        abl_loaded = None
                if abl_loaded is None:
                    print(f"[ABLATION] Running 5-subset feature ablation (seed={seeds[0]})...", flush=True)
                    abl_rows_run = run_feature_ablation(ds, seed=seeds[0], n_jobs=workers, fast_mode=fast_mode)
                    save_json_atomic(
                        abl_path,
                        {"seed": int(seeds[0]), "fast_mode": bool(fast_mode), "rows": abl_rows_run},
                    )

            for fam in families:
                for seed in seeds:
                    done_runs += 1
                    ckpt_path = ckpt_dir / f"{task.task_id}__{fam}__seed{seed}.json"
                    exp_meta = build_checkpoint_meta(ds, task.task_id, fam, seed, fast_mode)
                    if resume:
                        existing = load_valid_checkpoint(ckpt_path, exp_meta)
                        if existing is not None:
                            print(
                                f"[RESUME {done_runs}/{total_runs}] Skipping completed {ckpt_path.name}",
                                flush=True,
                            )
                            continue

                    t_pair = time.perf_counter()
                    print(
                        f"[RUN {done_runs}/{total_runs}] Starting {task.task_id} | {fam} | seed={seed} (n_jobs={workers})",
                        flush=True,
                    )
                    m, th = fit_select_and_eval(
                        ds,
                        fam,
                        seed,
                        binary=task.binary,
                        n_jobs=workers,
                        fast_mode=fast_mode,
                        task_id=task.task_id,
                    )
                    m["task_id"] = task.task_id
                    m["task_label"] = task.label
                    m["model"] = fam
                    m["seed"] = int(seed)
                    save_json_atomic(
                        ckpt_path,
                        {"meta": exp_meta, "metrics": m, "threshold_info": th},
                    )
                    print(
                        f"[DONE {done_runs}/{total_runs}] {ckpt_path.name} -> test_macro_f1={m['macro_f1']:.4f} "
                        f"({time.perf_counter() - t_pair:.2f}s)",
                        flush=True,
                    )

    # 2. Aggregate across the full expected matrix & explicitly flag incomplete experiments
    per_seed_all: list[dict[str, Any]] = []
    aggregated_all: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    threshold_summary: dict[str, Any] = {}
    incomplete_experiments: list[str] = []

    abl_path = ckpt_dir / f"feature_ablation__seed{seeds[0]}.json"
    ablation_rows: list[dict[str, Any]] = []
    if abl_path.exists():
        try:
            ablation_rows = json.loads(abl_path.read_text(encoding="utf-8")).get("rows", [])
        except Exception:
            ablation_rows = []

    for task in full_tasks:
        ds = _get_task_dataset(task, uad_dir, ds_cache)
        for fam in full_families:
            seed_runs: list[dict[str, Any]] = []
            thresh_runs: list[dict[str, Any]] = []
            missing_seeds: list[int] = []
            for seed in seeds:
                ckpt_path = ckpt_dir / f"{task.task_id}__{fam}__seed{seed}.json"
                exp_meta = build_checkpoint_meta(ds, task.task_id, fam, seed, fast_mode)
                loaded = load_valid_checkpoint(ckpt_path, exp_meta)
                if loaded is None:
                    missing_seeds.append(int(seed))
                else:
                    seed_runs.append(loaded["metrics"])
                    if loaded.get("threshold_info") is not None:
                        thresh_runs.append(loaded["threshold_info"])

            per_seed_all.extend(seed_runs)
            if missing_seeds:
                incomplete_experiments.append(f"{task.task_id}__{fam} (missing seeds: {missing_seeds})")
                continue

            agg_key = f"{task.task_id}__{fam}"
            agg = _aggregate_with_bal_acc(seed_runs, ds.class_names)
            agg["task_id"] = task.task_id
            agg["task_label"] = task.label
            agg["model"] = fam
            aggregated_all[agg_key] = agg

            cm_arr = np.asarray(agg["confusion_matrix_sum"], dtype=np.int64)
            save_confusion_matrix_plot(
                cm_arr,
                ds.class_names,
                f"{task.label} — {fam} (Sum over {len(seeds)} seeds)",
                cm_dir / f"{agg_key}.png",
            )
            save_json_atomic(
                cm_dir / f"{agg_key}.json",
                {
                    "class_names": list(ds.class_names),
                    "confusion_matrix_sum": agg["confusion_matrix_sum"],
                    "confusion_matrix_mean": agg["confusion_matrix_mean"],
                },
            )

            if thresh_runs:
                threshold_summary[agg_key] = {
                    "task_label": task.label,
                    "model": fam,
                    "tau_mean": float(np.mean([t["selected_threshold"] for t in thresh_runs])),
                    "default_f1_mean": float(np.mean([t["default_0.5_test_macro_f1"] for t in thresh_runs])),
                    "tuned_f1_mean": float(np.mean([t["tuned_test_macro_f1"] for t in thresh_runs])),
                    "default_p_mean": float(np.mean([t["default_0.5_attack_precision"] for t in thresh_runs])),
                    "default_r_mean": float(np.mean([t["default_0.5_attack_recall"] for t in thresh_runs])),
                    "default_0.5_attack_f1": float(np.mean([t["default_0.5_attack_f1"] for t in thresh_runs])),
                    "default_0.5_confusion_matrix": np.sum(
                        [np.asarray(t["default_0.5_confusion_matrix"], dtype=np.int64) for t in thresh_runs], axis=0
                    ).tolist(),
                    "tuned_p_mean": float(np.mean([t["tuned_attack_precision"] for t in thresh_runs])),
                    "tuned_r_mean": float(np.mean([t["tuned_attack_recall"] for t in thresh_runs])),
                    "tuned_attack_f1": float(np.mean([t["tuned_attack_f1"] for t in thresh_runs])),
                    "tuned_confusion_matrix": np.sum(
                        [np.asarray(t["tuned_confusion_matrix"], dtype=np.int64) for t in thresh_runs], axis=0
                    ).tolist(),
                }

            pc = agg["per_class"]
            comparison_rows.append(
                {
                    "task_id": task.task_id,
                    "task_label": task.label,
                    "model": fam,
                    "macro_f1_mean": agg["macro_f1_mean"],
                    "macro_f1_std": agg["macro_f1_std"],
                    "balanced_accuracy_mean": agg["balanced_accuracy_mean"],
                    "accuracy_mean": agg["accuracy_mean"],
                    "mitm_f1_mean": pc["mitm"]["f1_mean"] if "mitm" in pc else np.nan,
                    "intrusion_f1_mean": pc["intrusion"]["f1_mean"] if "intrusion" in pc else np.nan,
                    "train_time_s_mean": agg["train_time_s_mean"],
                    "inference_ms_per_1k_mean": agg["inference_ms_per_1k_mean"],
                }
            )

    config_info = {
        "seeds": list(seeds),
        "expected_tasks": [t.task_id for t in full_tasks],
        "expected_families": list(full_families),
        "requested_tasks": [t.task_id for t in tasks],
        "requested_families": list(families),
        "n_jobs": workers,
        "fast_mode": fast_mode,
        "resume": resume,
        "complete": len(incomplete_experiments) == 0,
        "incomplete_experiments": incomplete_experiments,
        "selection_criterion": "validation_macro_f1",
        "baseline_reference": str(baseline_csv),
    }
    save_json_atomic(out_dir / "experiment_config.json", config_info)
    save_json_atomic(out_dir / "per_seed_metrics.json", per_seed_all)
    save_json_atomic(out_dir / "aggregated_metrics.json", aggregated_all)
    save_json_atomic(out_dir / "threshold_tuning.json", threshold_summary)
    save_json_atomic(out_dir / "feature_ablation.json", ablation_rows)
    pd.DataFrame(ablation_rows).to_csv(out_dir / "feature_ablation.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(out_dir / "model_comparison.csv", index=False)

    generate_improvement_report(
        aggregated_all,
        ablation_rows,
        threshold_summary,
        baseline_csv,
        seeds,
        report_path,
        incomplete_experiments=incomplete_experiments,
    )
    if incomplete_experiments:
        print(f"[WARN] Incomplete experiments ({len(incomplete_experiments)}): {incomplete_experiments}", flush=True)
    return {
        "aggregated": aggregated_all,
        "ablation": ablation_rows,
        "thresholds": threshold_summary,
        "incomplete_experiments": incomplete_experiments,
    }


def _parse_csv_arg(val: str | None, valid: Sequence[str], kind: str) -> list[str]:
    if not val:
        return list(valid)
    items = [x.strip() for x in val.split(",") if x.strip()]
    bad = [x for x in items if x not in valid]
    if bad:
        raise ValueError(f"Unknown {kind}: {bad}. Valid options: {list(valid)}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3B.6 controlled baseline improvements.")
    parser.add_argument("--uad-dir", type=Path, default=DEFAULT_UAD_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated task IDs to run.")
    parser.add_argument("--families", type=str, default=None, help="Comma-separated model families to run.")
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs(), help="Bounded CPU workers (default: cpu_count-2).")
    parser.add_argument("--resume", action="store_true", help="Resume from valid checkpoints.")
    parser.add_argument("--aggregate-only", action="store_true", help="Aggregate existing checkpoints without training.")
    parser.add_argument("--fast", action="store_true", help="Use smaller grid for quick smoke testing only.")
    args = parser.parse_args()

    task_by_id = {t.task_id: t for t in UAD_TASKS}
    sel_task_ids = _parse_csv_arg(args.tasks, list(task_by_id.keys()), "tasks")
    sel_families = _parse_csv_arg(args.families, list(ALL_FAMILIES), "families")
    sel_tasks = tuple(task_by_id[tid] for tid in sel_task_ids)

    res = run_stage3b6(
        uad_dir=args.uad_dir,
        out_dir=args.out_dir,
        report_path=args.report_path,
        tasks=sel_tasks,
        families=sel_families,
        n_jobs=args.n_jobs,
        fast_mode=args.fast,
        resume=args.resume,
        aggregate_only=args.aggregate_only,
        expected_tasks=UAD_TASKS,
        expected_families=ALL_FAMILIES,
    )
    print(
        f"Stage 3B.6 aggregation complete ({len(res['aggregated'])} complete pairs, "
        f"{len(res['incomplete_experiments'])} incomplete) -> {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

