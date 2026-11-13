"""
Quick-start script: runs all steps that work WITHOUT needing
ML model training (tests, EFSMs, rule evaluation, unified diagnosis).

Use this while the heavy training is running in another terminal.

Run from project root:
  .venv\\Scripts\\python.exe run_quick_validation.py
"""
import subprocess
import sys
from pathlib import Path

PY   = sys.executable
ROOT = Path(__file__).resolve()

STEPS = [
    ("Unit tests",                   [PY, "-m", "pytest", "tests/", "-v"]),
    ("Temperature EFSM self-test",   [PY, "models_fsm/temperature_efsm.py"]),
    ("Motion EFSM self-test",        [PY, "models_fsm/motion_efsm.py"]),
    ("Unified diagnosis engine",     [PY, "diagnosis_loc/unified_diagnosis.py"]),
    ("Model comparison (any done)",  [PY, "ids/compare_models.py"]),
]

def run(label, cmd):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(ROOT.parent))
    if result.returncode == 0:
        print(f"[OK] {label}")
    else:
        print(f"[FAILED] {label} — exit code {result.returncode}")
    return result.returncode == 0

passed = []
failed = []
for label, cmd in STEPS:
    ok = run(label, cmd)
    (passed if ok else failed).append(label)

print(f"\n{'='*60}")
print(f"  Quick validation complete")
print(f"  Passed: {len(passed)}  Failed: {len(failed)}")
print(f"{'='*60}")
for s in passed:
    print(f"  [OK]     {s}")
for s in failed:
    print(f"  [FAILED] {s}")
