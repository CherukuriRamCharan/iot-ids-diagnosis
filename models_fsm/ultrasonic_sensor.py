"""Ultrasonic-sensor TFSM, its attack mutants and the mutation-derived test cases.

Formal Modeling & Mutants (Ultrasonic Sensor)
----------------------------------------------
Follows the timed-FSM formalization of Kaddoura et al. (Internet of Things 38, 2026).
Device role: Distance and proximity obstacle ranging (matches real-world topic 'distance/ultrasonic1').

Alphabets:
  inputs : i2 ws(1) | i4 cc(1) | i5 healthy-check | i6 unexpected-request | i7 stop-publishing
           i8 give-me-a-WiFi-name | i9 name-received | i10 start-publishing
           i11 DistanceEcho(clear_path) | i12 DistanceEcho(close_obstacle)
  outputs: o1 connecting-wifi | o2 connected-wifi | o3 connecting-MQTT | o4 connected-MQTT
           o5 sensor-alive | o6 error-invalid-message | o7 publishing-stopped | o8 press-any-button
           o10 publishing-started | o11 distance-reading-published | o12 proximity-alert-published
           o13 Device_SSID
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from .tfsm import (TFSM, Transition, add_state_after, mutate_output, mutate_timeout, parse)

DEVICE = "ultrasonic_sensor"
DIAG_CLASSES = ["Normal", "MITM", "Replay", "DataFalsification", "BatteryDraining"]


# ----------------------------------------------------------------- original machine
def build_original() -> TFSM:
    T: List[Transition] = [
        # start-up: wifi, then MQTT
        Transition("t1", "s1", "i2", "o2", "s2"),
        Transition("t2", "s1", None, "o1", "s1", 2.0),
        Transition("t8", "s2", "i4", "o4", "s5"),
        Transition("t3", "s2", None, "o3", "s2", 2.0),
        # waiting-for-WiFi-name / config state
        Transition("t15", "s6", "i9", "o13", "s5"),
        Transition("t13", "s6", "i6", "o6", "s6"),
        Transition("t16", "s6", None, "o8", "s6", 2.0),
        # timeouts of the operational states
        Transition("t19", "s5", None, "o5", "s8", 2.0),
        Transition("t18", "s7", None, "o5", "s8", 2.0),
        Transition("t20", "s8", None, "o5", "s9", 2.0),
        Transition("t22", "s9", None, "o5", "s9", 2.0),
    ]
    # operational states share the same logical transitions
    for s in ("s5", "s7", "s8", "s9"):
        T += [
            Transition("t5", s, "i5", "o5", s),       # health check
            Transition("t9", s, "i4", "o6", s),       # duplicate 'MQTT connected' is invalid
            Transition("t10", s, "i6", "o6", s),      # unexpected request
            Transition("t12", s, "i7", "o7", "s7"),   # stop publishing
            Transition("t14", s, "i8", "o8", "s6"),   # ask for WiFi name / config
            Transition("t17", s, "i10", "o10", "s7"), # start publishing
            Transition("t21", s, "i11", "o11", "s9"), # normal clear-path distance reading
            Transition("t23", s, "i12", "o12", "s9"), # close obstacle proximity alert event
        ]
    return TFSM("ultrasonic_orig", "s1", T)


# ------------------------------------------------------------------- test cases
TESTS: Dict[str, str] = {
    "A":    "Delay(2)-i2-delay(2)-i4-i7-delay(2)-i8-i9",              # DFA t8,t12,t14,t15
    "B":    "Delay(2)-i2-delay(2)-i4-i10-delay(2)-delay(2)-i11",      # DFA t17,t21
    "C":    "Delay(2)-i2-delay(2)-i4-i10-delay(2)-delay(2)-i12",      # DFA t23
    "MITM": "Delay(2)-i2-delay(2)-i4-delay(2)-i11-i11",
    "BD1":  "Delay(2)-i2-delay(2)-i4-delay(1)-i5",
    "BD2":  "Delay(2)-i2-delay(2)-i4-i8-delay(1)-i9",
    "BD3":  "Delay(2)-i2-delay(2)-i4-i10-delay(1)-i5",
    "BD4":  "Delay(2)-i2-delay(2)-i4-delay(2)-delay(1)-i11",
    "BD5":  "Delay(2)-i2-delay(2)-i4-delay(2)-i11-delay(1)-i11",
    "SDA":  "Delay(2)-i2-delay(2)-i4-delay(2)-i11",
    "RA":   "Delay(2)-i2-delay(2)-i4-i4",
}
TRAILING = 0.5


# ---------------------------------------------------------------------- mutants
@dataclass(frozen=True)
class MutantSpec:
    mid: str                 # localization label
    attack: str              # diagnosis label
    test: str                # key in TESTS: the test derived to kill it
    where: str               # human-readable fault location
    build: Callable[[TFSM], TFSM]


def _dfa(tid, where):
    tk = "A" if tid in ("t8", "t12", "t14", "t15") else ("C" if tid == "t23" else "B")
    return MutantSpec(f"DFA_{tid}", "DataFalsification", tk,
                      where, lambda m, tid=tid: mutate_output(m, tid, "o6", f"dfa_{tid}"))


MUTANTS: List[MutantSpec] = [
    _dfa("t8",  "transition t8  (s2 --i4/o4--> s5): 'MQTT connected' reply replaced by an error"),
    _dfa("t12", "transition t12 (i7/o7): 'publishing stopped' reply replaced by an error"),
    _dfa("t14", "transition t14 (i8/o8): 'press any button' reply replaced by an error"),
    _dfa("t15", "transition t15 (s6 --i9/o13--> s5): Device_SSID reply replaced by an error"),
    _dfa("t17", "transition t17 (i10/o10): 'publishing started' reply replaced by an error"),
    _dfa("t21", "transition t21 (i11/o11): distance measurement report replaced by an error"),
    _dfa("t23", "transition t23 (i12/o12): proximity obstacle alert report replaced by an error"),
    MutantSpec("MITM_s8", "MITM", "MITM",
               "extra state inserted after s8 (transition t21 from s8): distance echo packets are intercepted",
               lambda m: add_state_after(m, "t21", "s8", "mitm_s8")),
    MutantSpec("BD_s5", "BatteryDraining", "BD1", "timeout t19 of state s5 shortened 2s -> 1s",
               lambda m: mutate_timeout(m, "t19", 1.0, "bd_s5")),
    MutantSpec("BD_s6", "BatteryDraining", "BD2", "timeout t16 of state s6 shortened 2s -> 1s",
               lambda m: mutate_timeout(m, "t16", 1.0, "bd_s6")),
    MutantSpec("BD_s7", "BatteryDraining", "BD3", "timeout t18 of state s7 shortened 2s -> 1s",
               lambda m: mutate_timeout(m, "t18", 1.0, "bd_s7")),
    MutantSpec("BD_s8", "BatteryDraining", "BD4", "timeout t20 of state s8 shortened 2s -> 1s",
               lambda m: mutate_timeout(m, "t20", 1.0, "bd_s8")),
    MutantSpec("BD_s9", "BatteryDraining", "BD5", "timeout t22 of state s9 shortened 2s -> 1s",
               lambda m: mutate_timeout(m, "t22", 1.0, "bd_s9")),
    MutantSpec("BD_SDA_s9", "BatteryDraining", "SDA", "timeout t22 of state s9 shortened 2s -> 0.001s (sleep deprivation)",
               lambda m: mutate_timeout(m, "t22", 0.001, "bd_sda_s9")),
    MutantSpec("RA_t9", "Replay", "RA", "transition t9 (s5 --i4--> ): replayed 'MQTT connected' accepted (o6 -> o4)",
               lambda m: mutate_output(m, "t9", "o4", "ra_t9")),
]

LOC_CLASSES = ["normal"] + [m.mid for m in MUTANTS]
MUTANT_BY_ID = {m.mid: m for m in MUTANTS}


def steps_of(test_key: str):
    return parse(TESTS[test_key])
