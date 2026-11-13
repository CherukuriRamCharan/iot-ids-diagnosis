"""Turn model predictions into human-readable diagnosis + localization alerts.

    python -m diagnosis_loc.report --n 8
"""
import argparse
import csv

from models_fsm.motion_sensor import DEVICE, MUTANT_BY_ID


def alert(pred_diag: str, pred_loc: str) -> str:
    if pred_diag == "Normal" and pred_loc == "normal":
        return f"[{DEVICE}] OK - behaviour matches the specification."
    m = MUTANT_BY_ID.get(pred_loc)
    where = m.where if m else "unknown (diagnosis and localization disagree)"
    flag = "" if (m and m.attack == pred_diag) else "  (!) low-confidence: class/location mismatch"
    return f"[{DEVICE}] ALERT {pred_diag}\n    suspected fault: {where}{flag}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="results/test_predictions.csv")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    shown = 0
    for r in csv.DictReader(open(a.pred)):
        if r["pred_diag"] == "Normal" and shown >= a.n // 2:
            continue
        print(alert(r["pred_diag"], r["pred_loc"]))
        print(f"    ground truth: {r['true_diag']} / {r['true_loc']}\n")
        shown += 1
        if shown >= a.n:
            break


if __name__ == "__main__":
    main()
