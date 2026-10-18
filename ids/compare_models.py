"""
Model Comparison Evaluator
===========================
Reads evaluation JSON files from:
  - results/base_paper_model_evaluation.json   (binary RF)
  - results/rf_5class_evaluation.json          (5-class RF)
  - results/lstm_evaluation.json               (LSTM)
  - results/transformer_evaluation.json        (Transformer)

Produces:
  results/model_comparison.json   — machine-readable comparison table
  results/model_comparison.csv    — per-class metrics all models

Usage:
  .venv\\Scripts\\python.exe ids\\compare_models.py

Note: Only reads files that exist — safely skips missing models.
Results are ONLY comparable if:
  - Same test split was used
  - Same label definitions were used
Binary RF uses a different task (2 classes) from 5-class models.
Do NOT directly compare binary-RF accuracy to 5-class accuracy.
"""

import json
from pathlib import Path

import pandas as pd

ROOT       = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "results"

MODEL_FILES = {
    "RF_binary":     "base_paper_model_evaluation.json",
    "RF_5class":     "rf_5class_evaluation.json",
    "LSTM":          "lstm_evaluation.json",
    "Transformer":   "transformer_evaluation.json",
}

OUT_JSON = RESULT_DIR / "model_comparison.json"
OUT_CSV  = RESULT_DIR / "model_comparison.csv"


def load_eval(fname: str) -> dict | None:
    path = RESULT_DIR / fname
    if not path.exists():
        print(f"  [SKIP] {fname} — not found (run training first)")
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"  [OK]   {fname}")
    return data


def extract_summary(name: str, data: dict) -> dict:
    """Pull key metrics into a flat summary dict."""
    cr  = data.get("classification_report", {})
    acc = cr.get("accuracy", data.get("accuracy", None))

    # Macro-average F1
    macro = cr.get("macro avg", {})
    macro_f1  = macro.get("f1-score", None)
    macro_p   = macro.get("precision", None)
    macro_r   = macro.get("recall", None)

    # Weighted-average F1
    weighted = cr.get("weighted avg", {})
    weighted_f1 = weighted.get("f1-score", None)

    # Per-class rows
    classes = [k for k in cr if k not in ("accuracy", "macro avg", "weighted avg")]
    per_class = {}
    for cls in classes:
        per_class[cls] = {
            "precision": cr[cls].get("precision"),
            "recall":    cr[cls].get("recall"),
            "f1":        cr[cls].get("f1-score"),
            "support":   cr[cls].get("support"),
        }

    task      = data.get("task", "")
    features  = data.get("features", data.get("features_used", []))
    n_feat    = len(features) if isinstance(features, list) else 0
    split     = data.get("split_method", "")
    limits    = data.get("limitations", [])

    # Training history (LSTM / Transformer)
    history   = data.get("training_history", None)
    if isinstance(history, dict):
        best_val_loss = min(history.get("val_loss", [float("inf")]))
        best_val_acc  = max(history.get("val_acc", [0.0]))
    else:
        best_val_loss = None
        best_val_acc  = None

    return {
        "model":          name,
        "task":           task,
        "n_features":     n_feat,
        "accuracy":       round(acc, 4) if acc is not None else None,
        "macro_f1":       round(macro_f1, 4) if macro_f1 is not None else None,
        "macro_precision":round(macro_p,  4) if macro_p  is not None else None,
        "macro_recall":   round(macro_r,  4) if macro_r  is not None else None,
        "weighted_f1":    round(weighted_f1, 4) if weighted_f1 is not None else None,
        "split_method":   split,
        "best_val_loss":  round(best_val_loss, 5) if best_val_loss is not None else None,
        "best_val_acc":   round(best_val_acc,  4) if best_val_acc  is not None else None,
        "per_class":      per_class,
        "limitations":    limits,
    }


def main():
    print("=" * 60)
    print("  Model Comparison Evaluator")
    print("=" * 60)
    print("\nLoading model evaluation files …")

    summaries = []
    per_class_rows = []

    for name, fname in MODEL_FILES.items():
        data = load_eval(fname)
        if data is None:
            continue
        s = extract_summary(name, data)
        summaries.append(s)

        for cls, metrics in s["per_class"].items():
            per_class_rows.append({
                "model":     name,
                "class":     cls,
                "precision": metrics["precision"],
                "recall":    metrics["recall"],
                "f1":        metrics["f1"],
                "support":   metrics["support"],
            })

    if not summaries:
        print("\nNo evaluation files found. Run training scripts first.")
        return

    # ── Print comparison table ──
    print("\n" + "=" * 60)
    print("  SUMMARY COMPARISON")
    print("=" * 60)
    print(f"\n{'Model':<16} {'Task':<45} {'Acc':>6} {'MacroF1':>9} {'WtF1':>7}")
    print("-" * 88)
    for s in summaries:
        task_short = s["task"][:43] if s["task"] else "—"
        acc  = f"{s['accuracy']:.4f}" if s["accuracy"] is not None else "—"
        mf1  = f"{s['macro_f1']:.4f}"  if s["macro_f1"]  is not None else "—"
        wf1  = f"{s['weighted_f1']:.4f}" if s["weighted_f1"] is not None else "—"
        print(f"{s['model']:<16} {task_short:<45} {acc:>6} {mf1:>9} {wf1:>7}")

    print()
    print("⚠  Binary RF and 5-class models use different label sets.")
    print("   Do NOT interpret their accuracy scores as directly comparable.")
    print("   Binary RF: benign vs attack (2 classes)")
    print("   5-class:   benign/dos/mirai/mitm/scan (5 classes)")

    # ── Per-class table ──
    if per_class_rows:
        print("\n" + "=" * 60)
        print("  PER-CLASS METRICS")
        print("=" * 60)
        pc_df = pd.DataFrame(per_class_rows)
        print(pc_df.to_string(index=False))

    # ── Limitations ──
    print("\n" + "=" * 60)
    print("  LIMITATIONS (from each model)")
    print("=" * 60)
    for s in summaries:
        print(f"\n  {s['model']}:")
        for lim in s["limitations"]:
            print(f"    - {lim}")

    # ── Save ──
    RESULT_DIR.mkdir(exist_ok=True)

    comparison = {
        "note": (
            "Accuracy scores from different models are not directly comparable. "
            "Binary RF uses a 2-class task; 5-class models use a 5-class task. "
            "Test splits differ in size and composition."
        ),
        "models": summaries,
    }
    OUT_JSON.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    if per_class_rows:
        pd.DataFrame(per_class_rows).to_csv(OUT_CSV, index=False)
        print(f"\nSaved per-class CSV: {OUT_CSV}")

    print(f"Saved comparison JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
