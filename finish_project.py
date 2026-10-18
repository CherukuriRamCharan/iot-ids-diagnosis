"""
FINISH THE PROJECT — run all remaining steps automatically.

Usage (paste this ONE line into your VS Code terminal):

    .venv\\Scripts\\python.exe finish_project.py

This script will:
  1. Generate IAT features (~20-30 min)
  2. Train 5-class Random Forest (~5-10 min)
  3. Train LSTM (~15-30 min)
  4. Train Transformer (~15-30 min)
  5. Run unified diagnosis engine (~10 sec)
  6. Compare all models (~10 sec)
  7. Run all unit tests (~30 sec)

Total: ~1 to 1.5 hours. It prints progress as it goes.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY   = sys.executable

STEPS = [
    # (label, script_path_relative_to_ROOT, args)
    ("Step 1/7: Generate IAT features (20-30 min)",
     "data/base_paper/add_iat_features.py", []),

    ("Step 2/7: Train 5-class Random Forest (5-10 min)",
     "ids/train_rf_5class.py", []),

    ("Step 3/7: Train LSTM (15-30 min)",
     "ids/train_lstm.py", []),

    ("Step 4/7: Train Transformer (15-30 min)",
     "ids/train_transformer.py", []),

    ("Step 5/7: Run unified diagnosis engine",
     "diagnosis_loc/unified_diagnosis.py", []),

    ("Step 6/7: Compare all models",
     "ids/compare_models.py", []),
]


def run_step(label, script, args):
    full_path = ROOT / script
    if not full_path.exists():
        print(f"  [SKIP] {script} — file not found")
        return False

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Script: {script}")
    print(f"{'='*60}", flush=True)

    start = time.time()
    result = subprocess.run(
        [PY, str(full_path)] + args,
        cwd=str(ROOT),
    )
    elapsed = time.time() - start
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)

    if result.returncode == 0:
        print(f"\n  ✓ PASSED — {label} ({mins}m {secs}s)")
        return True
    else:
        print(f"\n  ✗ FAILED — {label} (exit code {result.returncode}, {mins}m {secs}s)")
        return False


def run_tests():
    print(f"\n{'='*60}")
    print(f"  Step 7/7: Run all unit tests")
    print(f"{'='*60}", flush=True)
    result = subprocess.run(
        [PY, "-m", "pytest", str(ROOT / "tests"), "-v"],
        cwd=str(ROOT),
    )
    if result.returncode == 0:
        print("\n  ✓ ALL TESTS PASSED")
        return True
    else:
        print(f"\n  ✗ SOME TESTS FAILED (exit code {result.returncode})")
        return False


def check_outputs():
    """Check which output files now exist."""
    expected = {
        "IAT CSV":                "data/base_paper/all_packet_features_iat.csv",
        "5-class RF model":       "models/rf_5class.joblib",
        "5-class RF evaluation":  "results/rf_5class_evaluation.json",
        "LSTM model":             "models/lstm_ids.pt",
        "LSTM evaluation":        "results/lstm_evaluation.json",
        "Transformer model":      "models/transformer_ids.pt",
        "Transformer evaluation": "results/transformer_evaluation.json",
        "Model comparison":       "results/model_comparison.json",
        "Unified diagnosis JSON": "results/unified_diagnosis_records.json",
        "Unified diagnosis CSV":  "results/unified_diagnosis_records.csv",
        # Pre-existing (should still exist):
        "Binary RF model":        "models/base_paper_rf.joblib",
        "Binary RF evaluation":   "results/base_paper_model_evaluation.json",
        "Combined diagnosis":     "results/combined_attack_diagnosis.csv",
        "MQTT rule report":       "results/combined_diagnosis_report.csv",
        "Capture-level eval":     "results/final_capture_evaluation.json",
    }

    print(f"\n{'='*60}")
    print(f"  OUTPUT FILE CHECK")
    print(f"{'='*60}")

    all_ok = True
    for label, rel_path in expected.items():
        p = ROOT / rel_path
        if p.exists():
            size = p.stat().st_size
            if size > 1024*1024:
                size_str = f"{size/1024/1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size} bytes"
            print(f"  ✓ {label:<30} {size_str}")
        else:
            print(f"  ✗ {label:<30} MISSING")
            all_ok = False

    return all_ok


def main():
    overall_start = time.time()

    print("="*60)
    print("  FINISH THE PROJECT — Automated Pipeline")
    print(f"  Project root: {ROOT}")
    print(f"  Python:       {PY}")
    print("="*60)

    passed = []
    failed = []

    for label, script, args in STEPS:
        ok = run_step(label, script, args)
        (passed if ok else failed).append(label)

    # Tests
    test_ok = run_tests()
    (passed if test_ok else failed).append("Unit tests")

    # Check outputs
    all_outputs = check_outputs()

    # Final summary
    total_time = time.time() - overall_start
    total_mins = int(total_time // 60)
    total_secs = int(total_time % 60)

    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Total time:  {total_mins}m {total_secs}s")
    print(f"  Steps passed: {len(passed)}/{len(passed)+len(failed)}")

    for s in passed:
        print(f"    ✓ {s}")
    for s in failed:
        print(f"    ✗ {s}")

    if all_outputs and not failed:
        print(f"\n  ★★★ PROJECT IS COMPLETE ★★★")
        print(f"  All models trained, all evaluations saved, all tests pass.")
        print(f"  Check the results/ folder for all output files.")
    elif not failed:
        print(f"\n  ★★ PROJECT IS FUNCTIONALLY COMPLETE ★★")
        print(f"  All steps passed. Some output files may be missing.")
    else:
        print(f"\n  ⚠ PROJECT HAS {len(failed)} FAILED STEP(S)")
        print(f"  Review the errors above and re-run failed steps manually.")

    print()


if __name__ == "__main__":
    main()
