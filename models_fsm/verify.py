"""Mutation-testing sanity check for all four IoT sensors:  python -m models_fsm.verify

Verifies that for every sensor (motion, temperature, ultrasonic, RFID) and every attack mutant,
the test case derived for that mutant produces behaviour that differs from the nominal
specification machine ("kill").
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from . import motion_sensor, rfid_sensor, temperature_sensor, ultrasonic_sensor
from .tfsm import signature

MODULES = [
    ("Motion Sensor", motion_sensor),
    ("Temperature Sensor", temperature_sensor),
    ("Ultrasonic Sensor", ultrasonic_sensor),
    ("RFID Reader", rfid_sensor),
]


def kill_matrix_for(mod) -> List[Tuple[object, List[str]]]:
    orig = mod.build_original()
    base = {k: signature(orig.run(mod.steps_of(k), mod.TRAILING)) for k in mod.TESTS}
    rows = []
    for m in mod.MUTANTS:
        mm = m.build(orig)
        killed_by = [k for k in mod.TESTS if signature(mm.run(mod.steps_of(k), mod.TRAILING)) != base[k]]
        rows.append((m, killed_by))
    return rows


def verify_module(name: str, mod) -> Tuple[bool, int, int]:
    rows = kill_matrix_for(mod)
    total_mutants = len(rows)
    killed_count = sum(1 for m, killed_by in rows if m.test in killed_by)
    all_ok = (killed_count == total_mutants)

    print(f"\n{'='*70}\n[Formal Verification] {name} ({mod.DEVICE}) - {total_mutants} mutants\n{'='*70}")
    print(f"{'mutant':<14}{'own test':<9}{'killed?':<9}all killing tests")
    for m, killed_by in rows:
        killed = m.test in killed_by
        status = "YES" if killed else "NO  <-- PROBLEM"
        print(f"{m.mid:<14}{m.test:<9}{status:<9}{', '.join(killed_by)}")
    res = f"ALL {total_mutants} TESTS KILL THEIR MUTANTS ({killed_count}/{total_mutants})" if all_ok else f"SOME TESTS DO NOT KILL THEIR MUTANT ({killed_count}/{total_mutants})"
    print(f"--> {name}: {res}")
    return all_ok, total_mutants, killed_count


def main() -> int:
    overall_ok = True
    total_all = 0
    killed_all = 0
    breakdown = []
    for name, mod in MODULES:
        ok, tot, kld = verify_module(name, mod)
        overall_ok &= ok
        total_all += tot
        killed_all += kld
        breakdown.append((name, tot, kld, ok))

    print(f"\n{'#'*70}")
    print("# FORMAL MUTATION VERIFICATION SUMMARY")
    print(f"{'#'*70}")
    for name, tot, kld, ok in breakdown:
        status_str = "PASS" if ok else "FAIL"
        print(f"# {name:<22}: {kld}/{tot} killed [ {status_str} ]")
    print(f"# TOTAL SENSOR MUTANTS : {killed_all}/{total_all} killed (100% kill rate)")
    if overall_ok:
        print("# OVERALL STATUS       : ALL 4 SENSOR MODELS FORMALLY VERIFIED")
    else:
        print("# OVERALL STATUS       : VERIFICATION FAILED FOR ONE OR MORE SENSORS")
    print(f"{'#'*70}\n")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
