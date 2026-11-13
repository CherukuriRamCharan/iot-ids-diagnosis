"""Turns a TFSM test run into a noisy 'network capture' (list of packets).

Noise model (per session / network profile):
  * delay jitter       : actual delay = nominal + |N(0, jitter)|  (so timeouts still fire)
  * tester latency     : U(lat_lo, lat_hi) seconds before every input
  * response delay     : device replies arrive Exp(resp_delay) s after the request
  * packet loss / dup  : independent per packet
  * payload size noise : multiplicative Gaussian
Each *session* has its own profile, so splitting train/test BY SESSION tests generalisation to
unseen network conditions (and avoids near-duplicate leakage between train and test).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from models_fsm.tfsm import TFSM, Step

# nominal MQTT payload sizes in bytes (o6 error messages are short, SSID replies longer)
BASE_SIZE: Dict[str, float] = {
    "i2": 48, "i4": 52, "i5": 40, "i6": 44, "i7": 46, "i8": 50, "i9": 58, "i10": 46, "i11": 64,
    "o1": 60, "o2": 62, "o3": 60, "o4": 64, "o5": 56, "o6": 36, "o7": 66, "o8": 70, "o10": 72,
    "o11": 78, "o12": 90,
}


@dataclass(frozen=True)
class NetProfile:
    jitter: float
    lat_lo: float
    lat_hi: float
    resp_delay: float
    loss: float
    dup: float
    size_noise: float


def make_profiles(n: int, rng: np.random.Generator) -> List[NetProfile]:
    profs = []
    for _ in range(n):
        lo = rng.uniform(0.02, 0.05)
        profs.append(NetProfile(
            jitter=rng.uniform(0.005, 0.06), lat_lo=lo, lat_hi=lo + rng.uniform(0.02, 0.08),
            resp_delay=rng.uniform(0.003, 0.04), loss=rng.uniform(0.0, 0.04),
            dup=rng.uniform(0.0, 0.03), size_noise=rng.uniform(0.02, 0.10)))
    return profs


def simulate(machine: TFSM, steps: List[Step], prof: NetProfile, rng: np.random.Generator,
             trailing: float = 0.5) -> List[dict]:
    """Run `steps` on `machine` under network profile `prof`; return packets sorted by time."""
    noisy: List[Step] = []
    for kind, val in steps:
        if kind == "delay":
            noisy.append(("delay", float(val) + abs(rng.normal(0, prof.jitter))))
        else:
            noisy.append(("delay", rng.uniform(prof.lat_lo, prof.lat_hi)))
            noisy.append(("in", val))
    events = machine.run(noisy, trailing=trailing)

    pkts: List[dict] = []
    last_out = 0.0
    for e in events:
        t = e.t
        if e.direction == "out":
            t = max(t + 0.002 + rng.exponential(prof.resp_delay), last_out)
            last_out = t
        if rng.random() < prof.loss:                      # packet lost
            continue
        size = BASE_SIZE.get(e.sym, 60.0) * (1 + rng.normal(0, prof.size_noise))
        pkts.append({"t": t, "dir": 0 if e.direction == "in" else 1, "sym": e.sym, "size": max(size, 8.0)})
        if rng.random() < prof.dup:                       # retransmission
            pkts.append({"t": t + rng.uniform(0.001, 0.01), "dir": pkts[-1]["dir"], "sym": e.sym,
                         "size": pkts[-1]["size"]})
    pkts.sort(key=lambda p: p["t"])
    return pkts
