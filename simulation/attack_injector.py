"""Sensor-Aware IoT Attack Injection Engine.

Injects controlled, observable attack behaviors into the UnifiedSimulator event stream:
1. Denial of Service (DoS): Rapid flooding of invalid/flood packets to exhaust receiver.
2. Replay: Capturing authentic historical telemetry and re-injecting with stale sequence/payload.
3. Man-in-the-Middle (MITM): In-flight payload manipulation and transmission interception.
4. Data Falsification (DFA): Out-of-bounds physical sensor falsification or output corruption.
5. Message Delay: Artificial buffering exceeding TFSM timing constraints (dt > timeout).
6. RFID Identity Spoofing: Forged unauthorized credential accepted as authorized.

Integrity & Anti-Leakage Constraints:
- Observable fields are modified or injected realistically.
- Ground truth metadata (is_attack, attack_type, attack_target, fault_location) is recorded
  strictly in the event's quarantined ground-truth partition.
- Original payload and pre-delay timestamps are preserved for auditability.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .unified_simulator import UnifiedEvent


@dataclass
class AttackSpec:
    """Specification of an attack to be injected into the simulation timeline."""

    attack_type: str                   # 'DoS' | 'Replay' | 'MITM' | 'DataFalsification' | 'Delay' | 'IdentitySpoofing'
    target_sensor: str                 # 'motion' | 'temperature' | 'ultrasonic' | 'rfid'
    start_time: float                  # Start of attack interval in simulation seconds
    duration: float                    # Duration of attack interval in seconds
    intensity: float = 1.0             # Rate or scaling multiplier
    params: Optional[Dict[str, Any]] = None  # Attack-specific parameter overrides


class AttackInjector:
    """Applies sensor-aware attack scenarios to a stream of UnifiedEvent objects."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def inject(
        self,
        events: List[UnifiedEvent],
        specs: List[AttackSpec],
    ) -> List[UnifiedEvent]:
        """Injects a list of AttackSpec scenarios into the event timeline."""
        # Deepcopy events to ensure source stream remains pristine
        modified_events = [copy.deepcopy(e) for e in events]

        for spec in specs:
            end_time = spec.start_time + spec.duration

            if spec.attack_type == "DoS":
                modified_events = self._inject_dos(modified_events, spec, end_time)
            elif spec.attack_type == "Replay":
                modified_events = self._inject_replay(modified_events, spec, end_time)
            elif spec.attack_type == "MITM":
                modified_events = self._inject_mitm(modified_events, spec, end_time)
            elif spec.attack_type == "DataFalsification":
                modified_events = self._inject_dfa(modified_events, spec, end_time)
            elif spec.attack_type == "Delay":
                modified_events = self._inject_delay(modified_events, spec, end_time)
            elif spec.attack_type == "IdentitySpoofing":
                modified_events = self._inject_spoofing(modified_events, spec, end_time)
            else:
                raise ValueError(f"Unknown attack type: {spec.attack_type}")

        # Re-sort all events chronologically
        modified_events.sort(key=lambda e: (e.timestamp, e.sequence_number))

        # Recompute inter-arrival deltas globally
        prev_t = 0.0
        for ev in modified_events:
            ev.inter_arrival = round(max(ev.timestamp - prev_t, 0.0), 4)
            prev_t = ev.timestamp

        return modified_events

    # ------------------------------------------------------------------
    # 1. Denial of Service (DoS Flood)
    # ------------------------------------------------------------------
    def _inject_dos(
        self,
        events: List[UnifiedEvent],
        spec: AttackSpec,
        end_time: float,
    ) -> List[UnifiedEvent]:
        flood_packets: List[UnifiedEvent] = []
        target = spec.target_sensor
        target_topic = f"home/sensor/{target}1" if target != "rfid" and target != "ultrasonic" else ("distance/ultrasonic1" if target == "ultrasonic" else "security/rfid1")

        # Determine flood frequency: e.g. 20 to 50 packets/sec
        rate = max(int(20 * spec.intensity), 5)
        dt = 1.0 / rate

        # Find latest sequence number for target sensor at start of attack
        seq_start = 1000
        for e in events:
            if e.sensor_id == target and e.timestamp <= spec.start_time:
                seq_start = max(seq_start, e.sequence_number)

        curr_t = spec.start_time + 0.001
        flood_seq = seq_start + 1
        while curr_t < end_time:
            flood_packets.append(
                UnifiedEvent(
                    timestamp=round(curr_t, 4),
                    sensor_id=target,
                    topic=target_topic,
                    event_type="FLOOD",
                    payload={"flood_payload": "X" * 128, "malformed": True},
                    sequence_number=flood_seq,
                    qos=0,
                    payload_size=164,
                    observed_state="s_busy",
                    output_symbol="o6",  # Receiver emits error on burst flood
                    is_attack=1,
                    attack_type="DoS",
                    attack_target=target,
                    fault_location="dos_queue_exhaustion",
                )
            )
            flood_seq += 1
            curr_t += dt + float(self.rng.uniform(-0.002, 0.002))

        return events + flood_packets

    # ------------------------------------------------------------------
    # 2. Replay Attack
    # ------------------------------------------------------------------
    def _inject_replay(
        self,
        events: List[UnifiedEvent],
        spec: AttackSpec,
        end_time: float,
    ) -> List[UnifiedEvent]:
        replayed_packets: List[UnifiedEvent] = []
        delay_offset = spec.params.get("delay_offset", 5.0) if spec.params else 5.0

        # Candidate packets occurring before the attack window to replay
        candidates = [
            e for e in events
            if e.sensor_id == spec.target_sensor
            and e.is_attack == 0
            and e.timestamp < spec.start_time
        ]

        if not candidates:
            return events

        # Select candidates and re-inject them with delayed timestamps during attack window
        for cand in candidates[-5:]:
            replay_t = round(cand.timestamp + delay_offset, 4)
            if spec.start_time <= replay_t <= end_time:
                rep = copy.deepcopy(cand)
                rep.timestamp = replay_t
                # Preserve the original sequence number (observable anomaly: non-monotonic / stale seq)
                rep.sequence_number = cand.sequence_number
                rep.output_symbol = "o6"  # Replayed message rejected or duplicate flag
                rep.is_attack = 1
                rep.attack_type = "Replay"
                rep.attack_target = spec.target_sensor
                rep.fault_location = "replayed_stale_sequence"
                rep.original_timestamp = cand.timestamp
                rep.original_payload = copy.deepcopy(cand.payload)
                replayed_packets.append(rep)

        return events + replayed_packets

    # ------------------------------------------------------------------
    # 3. Man-in-the-Middle (MITM Payload Tampering)
    # ------------------------------------------------------------------
    def _inject_mitm(
        self,
        events: List[UnifiedEvent],
        spec: AttackSpec,
        end_time: float,
    ) -> List[UnifiedEvent]:
        for e in events:
            if (
                e.sensor_id == spec.target_sensor
                and spec.start_time <= e.timestamp <= end_time
                and e.is_attack == 0
            ):
                orig_payload = copy.deepcopy(e.payload)
                # Tamper with domain payload
                if spec.target_sensor == "motion":
                    e.payload["motion"] = 1 - e.payload.get("motion", 0)
                    e.payload["tampered_by"] = "mitm_proxy"
                elif spec.target_sensor == "temperature":
                    e.payload["temperature"] = round(e.payload.get("temperature", 22.0) + 15.0, 2)
                    e.payload["tampered_by"] = "mitm_proxy"
                elif spec.target_sensor == "ultrasonic":
                    e.payload["distance_cm"] = 5.0  # Injected false obstacle
                    e.payload["obstacle"] = True
                elif spec.target_sensor == "rfid":
                    e.payload["status"] = "tampered"

                # Observable network effect: relay latency
                e.timestamp = round(e.timestamp + float(self.rng.uniform(0.04, 0.15)), 4)
                e.is_attack = 1
                e.attack_type = "MITM"
                e.attack_target = spec.target_sensor
                e.fault_location = "mitm_intercept_tamper"
                e.original_payload = orig_payload

        return events

    # ------------------------------------------------------------------
    # 4. Data Falsification (DFA Out-of-Bounds Manipulation)
    # ------------------------------------------------------------------
    def _inject_dfa(
        self,
        events: List[UnifiedEvent],
        spec: AttackSpec,
        end_time: float,
    ) -> List[UnifiedEvent]:
        for e in events:
            if (
                e.sensor_id == spec.target_sensor
                and spec.start_time <= e.timestamp <= end_time
                and e.is_attack == 0
            ):
                orig_payload = copy.deepcopy(e.payload)

                # Inject severe physical boundary violations
                if spec.target_sensor == "temperature":
                    e.payload["temperature"] = 185.5  # Extreme impossible temperature
                    e.payload["alert"] = True
                elif spec.target_sensor == "ultrasonic":
                    e.payload["distance_cm"] = -45.0  # Impossible negative distance
                elif spec.target_sensor == "motion":
                    e.payload["confidence"] = 999.0  # Impossible probability / confidence
                elif spec.target_sensor == "rfid":
                    e.payload["card_uid"] = "CORRUPTED_HEX_ZZ"

                e.output_symbol = "o6"  # Output fault
                e.is_attack = 1
                e.attack_type = "DataFalsification"
                e.attack_target = spec.target_sensor
                e.fault_location = "dfa_sensor_falsification"
                e.original_payload = orig_payload

        return events

    # ------------------------------------------------------------------
    # 5. Message Delay Attack (Exceeding TFSM Timeout)
    # ------------------------------------------------------------------
    def _inject_delay(
        self,
        events: List[UnifiedEvent],
        spec: AttackSpec,
        end_time: float,
    ) -> List[UnifiedEvent]:
        delay_sec = spec.params.get("delay_sec", 3.0) if spec.params else 3.0

        for e in events:
            if (
                e.sensor_id == spec.target_sensor
                and spec.start_time <= e.timestamp <= end_time
                and e.is_attack == 0
            ):
                orig_t = e.timestamp
                e.timestamp = round(e.timestamp + delay_sec, 4)
                e.is_attack = 1
                e.attack_type = "Delay"
                e.attack_target = spec.target_sensor
                e.fault_location = "delay_timeout_exceeded"
                e.original_timestamp = orig_t

        return events

    # ------------------------------------------------------------------
    # 6. RFID Identity Spoofing (Unauthorized UID Accepted)
    # ------------------------------------------------------------------
    def _inject_spoofing(
        self,
        events: List[UnifiedEvent],
        spec: AttackSpec,
        end_time: float,
    ) -> List[UnifiedEvent]:
        for e in events:
            if (
                e.sensor_id == "rfid"
                and spec.start_time <= e.timestamp <= end_time
                and e.is_attack == 0
            ):
                orig_payload = copy.deepcopy(e.payload)
                # Change unauthorized card or inject spoofed token
                e.payload["card_uid"] = "SPOOFED_UNAUTH_UID"
                e.payload["status"] = "authorized"  # Exploit: Unauthorized badge accepted
                e.output_symbol = "o11"  # Access granted output symbol (SPOOF_t23 mutant)
                e.observed_state = "s9"
                e.is_attack = 1
                e.attack_type = "IdentitySpoofing"
                e.attack_target = "rfid"
                e.fault_location = "rfid_uid_spoofing_t23"
                e.original_payload = orig_payload

        return events

