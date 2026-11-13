"""Minimal timed finite state machine (TFSM) engine + mutation operators.

Semantics (follows the timed-FSM description in the base paper):
  * a single clock, reset to 0 on EVERY transition;
  * a state may have one timeout transition: if no input arrives before the
    timeout expires, the timeout transition fires (and emits its output);
  * an input with no defined transition in the current state is answered with
    the error output "o6" and the state does not change.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Tuple

EPS = 1e-9
DEFAULT_ERR = "o6"
MAX_TIMEOUT_BURST = 8   # cap on timeout firings inside one delay (sleep-deprivation bursts)

Step = Tuple[str, object]          # ("delay", seconds) | ("in", "i4")


@dataclass(frozen=True)
class Transition:
    tid: str                       # logical transition id (shared ids = one logical transition)
    src: str
    inp: Optional[str]             # None => timeout transition
    out: str
    dst: str
    timeout: Optional[float] = None


@dataclass(frozen=True)
class Event:
    t: float                       # seconds since start of the run
    direction: str                 # "in"  = message sent to the device, "out" = sent by the device
    sym: str                       # i-symbol or o-symbol
    tid: str                       # transition that produced it ("-" for inputs)
    state: str                     # state after the event


class TFSM:
    def __init__(self, name: str, init: str, transitions):
        self.name = name
        self.init = init
        self.transitions = tuple(transitions)

    # ------------------------------------------------------------------ lookup
    def _timeout(self, state: str) -> Optional[Transition]:
        for tr in self.transitions:
            if tr.src == state and tr.inp is None:
                return tr
        return None

    def _on_input(self, state: str, inp: str) -> Optional[Transition]:
        for tr in self.transitions:
            if tr.src == state and tr.inp == inp:
                return tr
        return None

    # --------------------------------------------------------------------- run
    def run(self, steps: List[Step], trailing: float = 0.0) -> List[Event]:
        """Execute a test case (list of steps). `trailing` = idle observation time at the end."""
        st = {"state": self.init, "t": 0.0, "clock": 0.0}
        events: List[Event] = []

        def advance(d: float) -> None:
            remaining, fired = d, 0
            while True:
                tr = self._timeout(st["state"])
                if tr is not None and st["clock"] + remaining >= tr.timeout - EPS:
                    dt = max(tr.timeout - st["clock"], 0.0)
                    st["t"] += dt
                    remaining -= dt
                    st["state"], st["clock"] = tr.dst, 0.0
                    events.append(Event(st["t"], "out", tr.out, tr.tid, st["state"]))
                    fired += 1
                    if fired >= MAX_TIMEOUT_BURST:      # truncate runaway bursts
                        st["t"] += max(remaining, 0.0)
                        return
                else:
                    st["t"] += remaining
                    st["clock"] += remaining
                    return

        for kind, val in steps:
            if kind == "delay":
                advance(float(val))
            else:
                tr = self._on_input(st["state"], str(val))
                events.append(Event(st["t"], "in", str(val), "-", st["state"]))
                if tr is None:
                    out, dst, tid = DEFAULT_ERR, st["state"], "tx"
                else:
                    out, dst, tid = tr.out, tr.dst, tr.tid
                st["state"], st["clock"] = dst, 0.0
                events.append(Event(st["t"], "out", out, tid, dst))
        if trailing > 0:
            advance(trailing)
        return events


# ------------------------------------------------------------------- test syntax
def parse(seq: str) -> List[Step]:
    """'Delay(2)-i2-delay(1)-i5' -> [('delay',2.0),('in','i2'),('delay',1.0),('in','i5')]"""
    steps: List[Step] = []
    for tok in seq.split("-"):
        tok = tok.strip().lower()
        m = re.fullmatch(r"delay\(([\d.]+)\)", tok)
        if m:
            steps.append(("delay", float(m.group(1))))
        elif re.fullmatch(r"i\d+", tok):
            steps.append(("in", tok))
        else:
            raise ValueError(f"bad test token: {tok!r}")
    return steps


# -------------------------------------------------------------- mutation operators
def mutate_output(m: TFSM, tid: str, new_out: str, name: str) -> TFSM:
    """Output fault: change the output of logical transition `tid`."""
    trs = [replace(t, out=new_out) if t.tid == tid else t for t in m.transitions]
    return TFSM(name, m.init, trs)


def mutate_timeout(m: TFSM, tid: str, new_timeout: float, name: str) -> TFSM:
    """Timing fault: change the timeout value of timeout transition `tid`."""
    trs = [replace(t, timeout=new_timeout) if (t.tid == tid and t.inp is None) else t
           for t in m.transitions]
    return TFSM(name, m.init, trs)


def add_state_after(m: TFSM, tid: str, src: str, name: str, new_state: str = "sM") -> TFSM:
    """Man-in-the-middle: insert an extra state after transition `tid` leaving `src`.
    The new state has no outgoing transitions (every input is answered with the error output
    and it has no timeout), i.e. the device no longer behaves as specified afterwards."""
    trs = [replace(t, dst=new_state) if (t.tid == tid and t.src == src) else t
           for t in m.transitions]
    return TFSM(name, m.init, trs)


def signature(events: List[Event]) -> Tuple[str, ...]:
    """Observable behaviour used to decide whether a test kills a mutant."""
    return tuple(f"{e.direction}:{e.sym}" for e in events)
