"""Build the diagnosis + localization dataset for the motion sensor.

    python -m features.build_dataset --n-per-class 500 --sessions 10 --seed 0

Every sample = one execution of a mutation-derived test case against either the original
machine (label Normal) or one attack mutant. Three labels per sample:
    diag : Normal | MITM | Replay | DataFalsification | BatteryDraining     (attack diagnosis)
    loc  : normal | DFA_t8 | ... | RA_t9                                    (fault localization)
    session : network profile id (use it to split train/test, see ids/train.py)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from models_fsm.motion_sensor import (DEVICE, DIAG_CLASSES, LOC_CLASSES, MUTANTS, TESTS, TRAILING,
                                      build_original, steps_of)
from simulation.simulator import make_profiles, simulate

VOCAB = ["<pad>"] + [f"i{k}" for k in range(1, 13)] + [f"o{k}" for k in range(1, 17)]
TOK = {s: i for i, s in enumerate(VOCAB)}
MAXLEN = 24


def encode(pkts):
    tok = np.zeros(MAXLEN, dtype=np.int64)
    cont = np.zeros((MAXLEN, 3), dtype=np.float32)       # log1p(dt), size/100, direction
    prev = 0.0
    for k, p in enumerate(pkts[:MAXLEN]):
        tok[k] = TOK[p["sym"]]
        cont[k] = (np.log1p(max(p["t"] - prev, 0.0)), p["size"] / 100.0, p["dir"])
        prev = p["t"]
    return tok, cont


def build(n_per_class=500, n_sessions=10, seed=0):
    rng = np.random.default_rng(seed)
    profiles = make_profiles(n_sessions, rng)
    orig = build_original()

    # class -> list of (loc label, machine, test key)
    plan = defaultdict(list)
    for tk in TESTS:                                                   # Normal: all 10 tests on the original
        plan["Normal"].append(("normal", orig, tk))
    for m in MUTANTS:
        plan[m.attack].append((m.mid, m.build(orig), m.test))

    rows, toks, conts, y_diag, y_loc, sess = [], [], [], [], [], []
    for diag, variants in plan.items():
        base, extra = divmod(n_per_class, len(variants))
        for vi, (loc, machine, tk) in enumerate(variants):
            for k in range(base + (1 if vi < extra else 0)):
                s = k % n_sessions                                     # every variant appears in every session
                pk = simulate(machine, steps_of(tk), profiles[s], rng, TRAILING)
                tok, cont = encode(pk)
                toks.append(tok); conts.append(cont)
                y_diag.append(DIAG_CLASSES.index(diag)); y_loc.append(LOC_CLASSES.index(loc)); sess.append(s)
                rows.append({"device": DEVICE, "session": s, "diag": diag, "loc": loc, "test": tk,
                             "capture": " ".join(f"{p['sym']}@{p['t']:.3f}" for p in pk)})
    perm = rng.permutation(len(rows))
    arr = lambda x, dt=None: np.asarray(x, dtype=dt)[perm]
    data = dict(tok=arr(toks), cont=arr(conts), y_diag=arr(y_diag), y_loc=arr(y_loc), session=arr(sess))
    rows = [rows[i] for i in perm]
    return data, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=500)
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="features/data")
    a = ap.parse_args()

    data, rows = build(a.n_per_class, a.sessions, a.seed)
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "motion_sensor.npz"), **data)
    with open(os.path.join(a.out, "motion_sensor.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out, "labels.json"), "w") as f:
        json.dump({"diag": DIAG_CLASSES, "loc": LOC_CLASSES, "vocab": VOCAB, "maxlen": MAXLEN}, f, indent=1)
    print(f"samples: {len(rows)} | diag classes: {len(DIAG_CLASSES)} | loc classes: {len(LOC_CLASSES)}")
    for c in DIAG_CLASSES:
        print(f"  {c:<18}{sum(r['diag'] == c for r in rows)}")
    print("saved to", a.out)


if __name__ == "__main__":
    main()
