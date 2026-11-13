"""Stage 3B: Classical machine learning baselines (Random Forest & Decision Tree).

Trains and evaluates RandomForestClassifier and DecisionTreeClassifier on:
1. Primary benchmark: MQTT-UAD (multiclass & binary; full-stream & mqtt_only=True)
2. Secondary benchmark: MQTTset (multiclass & binary; kept strictly separate)

Uses Stage 3A leakage-safe loaders (ids/datasets_real.py) and saves per-seed
metrics, aggregated mean/std metrics, confusion matrices, model_comparison.csv,
and STAGE_3B_BASELINE_REPORT.md under results/stage_3b_baselines/.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier

from ids.datasets_real import (
    DEFAULT_MQTTSET_DIR,
    DEFAULT_UAD_DIR,
    SplitDataset,
    assert_no_leakage_columns,
    load_mqtt_uad,
    load_mqttset,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "results" / "stage_3b_baselines"
DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
DT_MAX_DEPTH_GRID: tuple[int | None, ...] = (8, 12, 16, 24, None)


@dataclass(frozen=True)
class TaskSpec:
    """Defines a single benchmark evaluation task."""

    task_id: str
    dataset_group: str  # "MQTT-UAD" or "MQTTset"
    dataset_label: str
    binary: bool
    mqtt_only: bool = False


DEFAULT_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec("mqtt_uad_full_multiclass", "MQTT-UAD", "MQTT-UAD (Full Stream, 4-Class)", binary=False, mqtt_only=False),
    TaskSpec("mqtt_uad_full_binary", "MQTT-UAD", "MQTT-UAD (Full Stream, Binary)", binary=True, mqtt_only=False),
    TaskSpec("mqtt_uad_mqtt_only_multiclass", "MQTT-UAD", "MQTT-UAD (MQTT-Only, 4-Class)", binary=False, mqtt_only=True),
    TaskSpec("mqtt_uad_mqtt_only_binary", "MQTT-UAD", "MQTT-UAD (MQTT-Only, Binary)", binary=True, mqtt_only=True),
    TaskSpec("mqttset_multiclass", "MQTTset", "MQTTset (Secondary Flow Benchmark, 6-Class)", binary=False),
    TaskSpec("mqttset_binary", "MQTTset", "MQTTset (Secondary Flow Benchmark, Binary)", binary=True),
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Computes full, macro, weighted, per-class metrics, and confusion matrix."""
    n_classes = len(class_names)
    labels = list(range(n_classes))

    acc = float(accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    weighted_p = float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))

    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_class: dict[str, dict[str, Any]] = {}
    absent_classes: list[str] = []
    for idx, cname in enumerate(class_names):
        sup = int(sup_cls[idx])
        if sup == 0:
            absent_classes.append(cname)
        per_class[cname] = {
            "precision": float(p_cls[idx]),
            "recall": float(r_cls[idx]),
            "f1": float(f1_cls[idx]),
            "support": sup,
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "absent_test_classes": absent_classes,
        "confusion_matrix": cm,
    }


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: Sequence[str],
    title: str,
    out_path: Path,
) -> None:
    """Saves a row-normalized confusion matrix heatmap."""
    row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm = cm / row_sums
    size = max(5.0, 0.7 * len(class_names) + 2.5)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}\n({int(cm[i, j])})",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def train_and_eval_decision_tree(
    ds: SplitDataset,
    seed: int,
    depth_grid: Sequence[int | None] = DT_MAX_DEPTH_GRID,
) -> dict[str, Any]:
    """Tunes max_depth on validation macro-F1 (train-only fit) and evaluates on test."""
    assert_no_leakage_columns(ds.feature_names)
    labels = list(range(len(ds.class_names)))

    best_depth = depth_grid[0]
    best_val_f1 = -1.0
    best_model: DecisionTreeClassifier | None = None
    t0 = time.perf_counter()

    for depth in depth_grid:
        clf = DecisionTreeClassifier(
            max_depth=depth,
            class_weight="balanced",
            random_state=seed,
        )
        clf.fit(ds.X_train, ds.y_train)
        val_pred = clf.predict(ds.X_val)
        val_f1 = float(f1_score(ds.y_val, val_pred, labels=labels, average="macro", zero_division=0))
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_depth = depth
            best_model = clf

    train_time_s = time.perf_counter() - t0
    assert best_model is not None

    t1 = time.perf_counter()
    test_pred = best_model.predict(ds.X_test)
    infer_time_s = time.perf_counter() - t1

    metrics = compute_classification_metrics(ds.y_test, test_pred, ds.class_names)
    metrics["best_val_macro_f1"] = best_val_f1
    metrics["selected_max_depth"] = best_depth
    metrics["train_time_s"] = train_time_s
    metrics["inference_time_s"] = infer_time_s
    metrics["inference_ms_per_1k"] = (1000.0 * infer_time_s / max(len(ds.X_test), 1)) * 1000.0
    return metrics


def train_and_eval_random_forest(
    ds: SplitDataset,
    seed: int,
    n_estimators: int = 100,
    max_depth: int | None = 20,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Trains RandomForestClassifier on X_train only and evaluates on val & test."""
    assert_no_leakage_columns(ds.feature_names)
    labels = list(range(len(ds.class_names)))

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
    )
    t0 = time.perf_counter()
    clf.fit(ds.X_train, ds.y_train)
    train_time_s = time.perf_counter() - t0

    val_pred = clf.predict(ds.X_val)
    val_f1 = float(f1_score(ds.y_val, val_pred, labels=labels, average="macro", zero_division=0))

    t1 = time.perf_counter()
    test_pred = clf.predict(ds.X_test)
    infer_time_s = time.perf_counter() - t1

    metrics = compute_classification_metrics(ds.y_test, test_pred, ds.class_names)
    metrics["val_macro_f1"] = val_f1
    metrics["n_estimators"] = n_estimators
    metrics["max_depth"] = max_depth
    metrics["train_time_s"] = train_time_s
    metrics["inference_time_s"] = infer_time_s
    metrics["inference_ms_per_1k"] = (1000.0 * infer_time_s / max(len(ds.X_test), 1)) * 1000.0
    return metrics


def aggregate_seed_runs(
    seed_runs: list[dict[str, Any]],
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Computes mean/std across seeds for scalar metrics and sums confusion matrices."""
    scalar_keys = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "train_time_s",
        "inference_time_s",
        "inference_ms_per_1k",
    )
    summary: dict[str, Any] = {"num_seeds": len(seed_runs)}
    for k in scalar_keys:
        vals = np.array([r[k] for r in seed_runs], dtype=np.float64)
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals, ddof=0))

    per_class_agg: dict[str, dict[str, Any]] = {}
    for cname in class_names:
        sup = int(seed_runs[0]["per_class"][cname]["support"])
        per_class_agg[cname] = {"support": sup}
        for m in ("precision", "recall", "f1"):
            vals = np.array([r["per_class"][cname][m] for r in seed_runs], dtype=np.float64)
            per_class_agg[cname][f"{m}_mean"] = float(np.mean(vals))
            per_class_agg[cname][f"{m}_std"] = float(np.std(vals, ddof=0))
    summary["per_class"] = per_class_agg
    summary["absent_test_classes"] = seed_runs[0]["absent_test_classes"]

    cm_sum = np.sum([np.asarray(r["confusion_matrix"], dtype=np.int64) for r in seed_runs], axis=0)
    cm_mean = np.mean([np.asarray(r["confusion_matrix"], dtype=np.float64) for r in seed_runs], axis=0)
    summary["confusion_matrix_sum"] = cm_sum.tolist()
    summary["confusion_matrix_mean"] = cm_mean.tolist()
    return summary


def _load_task_dataset(task: TaskSpec, uad_dir: Path, mqttset_dir: Path) -> SplitDataset:
    if task.dataset_group == "MQTT-UAD":
        return load_mqtt_uad(uad_dir, mqtt_only=task.mqtt_only, binary=task.binary)
    return load_mqttset(mqttset_dir, binary=task.binary)


def generate_markdown_report(
    aggregated: dict[str, Any],
    task_metadata: dict[str, Any],
    seeds: Sequence[int],
    out_path: Path,
) -> None:
    """Writes STAGE_3B_BASELINE_REPORT.md strictly from computed metrics."""
    lines: list[str] = [
        "# Stage 3B Classical Machine Learning Baseline Report",
        "",
        f"> **Seeds:** `{list(seeds)}` | **Primary Selection Metric:** `Macro-F1`",
        "> **Scope Note:** MQTT-UAD (within-capture chronological holdout) and MQTTset (secondary Flow_ID-grouped deduplicated benchmark) are evaluated strictly separately. Classification metrics do **not** prove cross-capture generalization, attack diagnosis, or formal fault localization.",
        "",
        "## 1. Experimental Setup & Safeguards",
        "- **Models:** `RandomForestClassifier` (`class_weight='balanced'`, `n_estimators=100`, `max_depth=20`) and `DecisionTreeClassifier` (`class_weight='balanced'`, `max_depth` tuned on validation `macro_f1`).",
        "- **Leakage Controls:** Loaded exclusively via Stage 3A `ids/datasets_real.py` (`assert_no_leakage_columns`). No metadata, timestamps, IPs, MACs, ports, topics, or client IDs were used.",
        "",
        "## 2. Model Comparison Summary (Mean ± Std Across Seeds)",
        "",
        "| Benchmark | Task | Model | Macro-F1 | Accuracy | Macro-P | Macro-R | Weighted-F1 | Train (s) | Infer (ms/1k) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for key, agg in aggregated.items():
        tmeta = task_metadata[agg["task_id"]]
        lines.append(
            f"| {tmeta['dataset_group']} | {tmeta['dataset_label']} | {agg['model']} | "
            f"{agg['macro_f1_mean']:.4f} ± {agg['macro_f1_std']:.4f} | "
            f"{agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f} | "
            f"{agg['macro_precision_mean']:.4f} | {agg['macro_recall_mean']:.4f} | "
            f"{agg['weighted_f1_mean']:.4f} | {agg['train_time_s_mean']:.2f} | "
            f"{agg['inference_ms_per_1k_mean']:.3f} |"
        )

    lines.extend(["", "## 3. Per-Class Test Metrics & Absent-Class Limitations", ""])
    for key, agg in aggregated.items():
        tmeta = task_metadata[agg["task_id"]]
        lines.append(f"### {tmeta['dataset_label']} — {agg['model']}")
        lines.append(
            f"- Train/Val/Test rows: `{tmeta['n_train']:,}` / `{tmeta['n_val']:,}` / `{tmeta['n_test']:,}` | Features: `{tmeta['n_features']}`"
        )
        if agg["absent_test_classes"]:
            lines.append(
                f"- **Absent Test Classes (support=0):** `{agg['absent_test_classes']}` (cannot be evaluated on this test split)."
            )
        lines.append("")
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
        "## 4. Key Limitations",
        "1. **MQTT-UAD:** Chronological splits are within single capture files; results do not establish unseen-capture generalization.",
        "2. **MQTTset:** Secondary deduplicated flow benchmark (`426,554` duplicates removed); `Flow_ID` grouping does not remove single-capture-per-`Cat` artifacts.",
        "3. **Scope:** Neither dataset contains formal TFSM fault-localization ground truth.",
    ])

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage3b(
    *,
    uad_dir: Path = DEFAULT_UAD_DIR,
    mqttset_dir: Path = DEFAULT_MQTTSET_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    tasks: Sequence[TaskSpec] = DEFAULT_TASKS,
    rf_estimators: int = 100,
    rf_max_depth: int | None = 20,
    rf_n_jobs: int = -1,
    dt_depth_grid: Sequence[int | None] = DT_MAX_DEPTH_GRID,
) -> dict[str, Any]:
    """Executes all Stage 3B baseline experiments and writes artifacts to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cm_dir = out_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    per_seed_all: list[dict[str, Any]] = []
    aggregated_all: dict[str, Any] = {}
    task_metadata: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []

    for task in tasks:
        ds = _load_task_dataset(task, uad_dir, mqttset_dir)
        assert_no_leakage_columns(ds.feature_names)
        task_metadata[task.task_id] = {
            "dataset_group": task.dataset_group,
            "dataset_label": task.dataset_label,
            "n_train": int(len(ds.X_train)),
            "n_val": int(len(ds.X_val)),
            "n_test": int(len(ds.X_test)),
            "n_features": int(len(ds.feature_names)),
            "class_names": list(ds.class_names),
        }

        for model_name in ("RandomForest", "DecisionTree"):
            seed_runs: list[dict[str, Any]] = []
            for seed in seeds:
                if model_name == "RandomForest":
                    m = train_and_eval_random_forest(
                        ds, seed, n_estimators=rf_estimators, max_depth=rf_max_depth, n_jobs=rf_n_jobs
                    )
                else:
                    m = train_and_eval_decision_tree(ds, seed, depth_grid=dt_depth_grid)
                m["task_id"] = task.task_id
                m["dataset_group"] = task.dataset_group
                m["model"] = model_name
                m["seed"] = int(seed)
                seed_runs.append(m)
                per_seed_all.append(m)

            agg_key = f"{task.task_id}__{model_name}"
            agg = aggregate_seed_runs(seed_runs, ds.class_names)
            agg["task_id"] = task.task_id
            agg["dataset_group"] = task.dataset_group
            agg["model"] = model_name
            aggregated_all[agg_key] = agg

            cm_arr = np.asarray(agg["confusion_matrix_sum"], dtype=np.int64)
            save_confusion_matrix_plot(
                cm_arr,
                ds.class_names,
                f"{task.dataset_label} — {model_name} (Sum over {len(seeds)} seeds)",
                cm_dir / f"{agg_key}.png",
            )
            (cm_dir / f"{agg_key}.json").write_text(
                json.dumps(
                    {
                        "class_names": list(ds.class_names),
                        "confusion_matrix_sum": agg["confusion_matrix_sum"],
                        "confusion_matrix_mean": agg["confusion_matrix_mean"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            comparison_rows.append(
                {
                    "dataset_group": task.dataset_group,
                    "task_id": task.task_id,
                    "dataset_label": task.dataset_label,
                    "model": model_name,
                    "n_train": int(len(ds.X_train)),
                    "n_val": int(len(ds.X_val)),
                    "n_test": int(len(ds.X_test)),
                    "n_features": int(len(ds.feature_names)),
                    "macro_f1_mean": agg["macro_f1_mean"],
                    "macro_f1_std": agg["macro_f1_std"],
                    "accuracy_mean": agg["accuracy_mean"],
                    "accuracy_std": agg["accuracy_std"],
                    "macro_precision_mean": agg["macro_precision_mean"],
                    "macro_recall_mean": agg["macro_recall_mean"],
                    "weighted_f1_mean": agg["weighted_f1_mean"],
                    "train_time_s_mean": agg["train_time_s_mean"],
                    "inference_ms_per_1k_mean": agg["inference_ms_per_1k_mean"],
                    "absent_test_classes": ",".join(agg["absent_test_classes"]),
                }
            )

    (out_dir / "per_seed_metrics.json").write_text(json.dumps(per_seed_all, indent=2), encoding="utf-8")
    (out_dir / "aggregated_metrics.json").write_text(json.dumps(aggregated_all, indent=2), encoding="utf-8")
    pd.DataFrame(comparison_rows).to_csv(out_dir / "model_comparison.csv", index=False)
    generate_markdown_report(aggregated_all, task_metadata, seeds, out_dir / "STAGE_3B_BASELINE_REPORT.md")
    return {"aggregated": aggregated_all, "task_metadata": task_metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3B classical ML baselines.")
    parser.add_argument("--uad-dir", type=Path, default=DEFAULT_UAD_DIR)
    parser.add_argument("--mqttset-dir", type=Path, default=DEFAULT_MQTTSET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rf-estimators", type=int, default=100)
    parser.add_argument("--rf-max-depth", type=int, default=20)
    args = parser.parse_args()

    res = run_stage3b(
        uad_dir=args.uad_dir,
        mqttset_dir=args.mqttset_dir,
        out_dir=args.out_dir,
        rf_estimators=args.rf_estimators,
        rf_max_depth=args.rf_max_depth,
    )
    print(f"Stage 3B baselines complete ({len(res['aggregated'])} model-task runs) -> {args.out_dir}")


if __name__ == "__main__":
    main()

