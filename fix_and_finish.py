"""
RECOVERY SCRIPT — fixes the 3 failed steps from finish_project.py.

Run this ONE command in your VS Code terminal:

    .venv\\Scripts\\python.exe fix_and_finish.py

It will:
  1. Install pytest (was missing)
  2. Evaluate saved LSTM model (--predict mode, no retraining)
  3. Evaluate saved Transformer model (--predict mode, no retraining)
  4. Re-run model comparison (now with all 4 models)
  5. Run all unit tests

Total time: ~5-10 minutes (no retraining needed).
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY   = sys.executable
PIP  = str(Path(PY).parent / "pip.exe")


def run(label, cmd, allow_fail=False):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}", flush=True)
    start = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - start
    ok = result.returncode == 0
    status = "PASSED" if ok else "FAILED"
    print(f"\n  [{status}] {label} ({int(elapsed)}s)")
    if not ok and not allow_fail:
        print(f"  Exit code: {result.returncode}")
    return ok


def main():
    print("="*60)
    print("  FIX & FINISH — Recovery for 3 failed steps")
    print("="*60)

    results = []

    # Step 1: Install pytest
    ok = run("Install pytest",
             [PY, "-m", "pip", "install", "pytest", "-q"])
    results.append(("Install pytest", ok))

    # Step 2: LSTM evaluation (--predict uses saved model, no retraining)
    lstm_model = ROOT / "models" / "lstm_ids.pt"
    if lstm_model.exists():
        ok = run("LSTM evaluation (using saved model)",
                 [PY, str(ROOT / "ids" / "train_lstm.py"), "--predict"])
        results.append(("LSTM evaluation", ok))
    else:
        # Model doesn't exist — need to retrain
        print("\n  [INFO] LSTM model not found — retraining from scratch...")
        ok = run("LSTM full training",
                 [PY, str(ROOT / "ids" / "train_lstm.py")])
        results.append(("LSTM training", ok))

    # Step 3: Transformer evaluation (--predict uses saved model)
    tf_model = ROOT / "models" / "transformer_ids.pt"
    if tf_model.exists():
        ok = run("Transformer evaluation (using saved model)",
                 [PY, str(ROOT / "ids" / "train_transformer.py"), "--predict"])
        results.append(("Transformer evaluation", ok))
    else:
        print("\n  [INFO] Transformer model not found — retraining from scratch...")
        ok = run("Transformer full training",
                 [PY, str(ROOT / "ids" / "train_transformer.py")])
        results.append(("Transformer training", ok))

    # Step 4: Re-run model comparison with all models
    ok = run("Model comparison (all 4 models)",
             [PY, str(ROOT / "ids" / "compare_models.py")])
    results.append(("Model comparison", ok))

    # Step 5: Run tests
    ok = run("Unit tests",
             [PY, "-m", "pytest", str(ROOT / "tests"), "-v"],
             allow_fail=True)
    results.append(("Unit tests", ok))

    # Final output check
    print(f"\n{'='*60}")
    print(f"  OUTPUT FILE CHECK")
    print(f"{'='*60}")
    expected = {
        "LSTM evaluation":        "results/lstm_evaluation.json",
        "Transformer evaluation": "results/transformer_evaluation.json",
        "Model comparison":       "results/model_comparison.json",
        "Unified diagnosis":      "results/unified_diagnosis_records.json",
        "Binary RF evaluation":   "results/base_paper_model_evaluation.json",
        "5-class RF evaluation":  "results/rf_5class_evaluation.json",
        "Combined diagnosis":     "results/combined_attack_diagnosis.csv",
    }
    all_exist = True
    for label, rel in expected.items():
        p = ROOT / rel
        if p.exists():
            sz = p.stat().st_size
            print(f"  ✓ {label:<30} ({sz/1024:.1f} KB)")
        else:
            print(f"  ✗ {label:<30} MISSING")
            all_exist = False

    # Summary
    print(f"\n{'='*60}")
    print(f"  FINAL RESULT")
    print(f"{'='*60}")
    for label, ok in results:
        print(f"  {'✓' if ok else '✗'} {label}")

    n_fail = sum(1 for _, ok in results if not ok)
    if n_fail == 0 and all_exist:
        print(f"\n  ★★★ PROJECT IS COMPLETE ★★★")
        print(f"  All models trained & evaluated.")
        print(f"  All output files present.")
        print(f"  All tests passing.")
    elif n_fail == 0:
        print(f"\n  ★★ STEPS PASSED but some output files missing ★★")
    else:
        print(f"\n  ⚠ {n_fail} step(s) still failing")
    print()


if __name__ == "__main__":
    main()
