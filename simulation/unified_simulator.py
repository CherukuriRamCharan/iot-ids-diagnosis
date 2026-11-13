"""Unified Four-Sensor IoT Event Simulator.

Simulates four independent IoT sensors (Motion, Temperature, Ultrasonic, RFID)
operating concurrently and communicating through a shared MQTT message layer.

Architecture:
- Shared monotonic virtual clock.
- Four independent TFSM instances (models_fsm/*.py) maintaining isolated state.
- Sensor-specific MQTT topics and common message schema.
- Deterministic simulation governed by explicit random seeds.
- Clean separation between observable message fields and ground-truth metadata.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models_fsm import motion_sensor, rfid_sensor, temperature_sensor, ultrasonic_sensor
from models_fsm.tfsm import Event, TFSM, Transition


# ----------------------------------------------------------------------
# Common MQTT Message Schema
# ----------------------------------------------------------------------
@dataclass
class UnifiedEvent:
    """Represents a single message or state transition event in the IoT system.

    Observability Rules:
    - OBSERVABLE FIELDS: Visible to the IDS, sequence models, and network monitors.
    - GROUND-TRUTH METADATA: Quarantined labels used strictly for evaluation,
      audit, and diagnosis ground truth. Never fed to predictive models.
    """

    # --- Observable Network & Protocol Fields ---
    timestamp: float                       # Monotonic virtual time (seconds)
    sensor_id: str                         # 'motion' | 'temperature' | 'ultrasonic' | 'rfid'
    topic: str                             # MQTT topic (e.g. 'distance/ultrasonic1')
    event_type: str                        # 'PUBLISH' | 'CONNECT' | 'HEARTBEAT' | 'ALERT'
    payload: Dict[str, Any]                # Physical telemetry / payload dictionary
    sequence_number: int                   # Monotonic per-sensor packet counter
    qos: int                               # MQTT QoS level (0 or 1)
    payload_size: int                      # Approximate byte size of the frame
    observed_state: str                    # Current TFSM state of the emitting sensor
    output_symbol: str                     # TFSM output symbol (e.g. 'o11', 'o5', 'o6')
    inter_arrival: float = 0.0             # Delta time from prior event on this sensor

    # --- Quarantined Ground-Truth Metadata (NEVER fed to model as input) ---
    is_attack: int = 0                     # Binary target: 0 = Benign, 1 = Attack
    attack_type: str = "Normal"            # 'Normal' | 'DoS' | 'Replay' | 'MITM' | 'DataFalsification' | 'Delay' | 'IdentitySpoofing'
    attack_target: str = "None"            # Target sensor id under attack
    fault_location: str = "None"           # Formal TFSM mutant / transition / state ID
    original_payload: Optional[Dict[str, Any]] = None  # Pre-tampered payload for audit
    original_timestamp: Optional[float] = None          # Pre-delay timestamp for audit

    def to_observable_dict(self) -> Dict[str, Any]:
        """Returns only features observable by the detection system."""
        return {
            "timestamp": self.timestamp,
            "sensor_id": self.sensor_id,
            "topic": self.topic,
            "event_type": self.event_type,
            "payload": copy.deepcopy(self.payload),
            "sequence_number": self.sequence_number,
            "qos": self.qos,
            "payload_size": self.payload_size,
            "observed_state": self.observed_state,
            "output_symbol": self.output_symbol,
            "inter_arrival": self.inter_arrival,
        }

    def to_ground_truth_dict(self) -> Dict[str, Any]:
        """Returns quarantined ground truth labels and audit traceability data."""
        return {
            "timestamp": self.timestamp,
            "sensor_id": self.sensor_id,
            "sequence_number": self.sequence_number,
            "is_attack": self.is_attack,
            "attack_type": self.attack_type,
            "attack_target": self.attack_target,
            "fault_location": self.fault_location,
            "original_payload": copy.deepcopy(self.original_payload),
            "original_timestamp": self.original_timestamp,
        }


# ----------------------------------------------------------------------
# Sensor State Wrappers
# ----------------------------------------------------------------------
class SensorRuntime:
    """Manages the lifecycle and TFSM state transitions of a single IoT sensor."""

    def __init__(
        self,
        sensor_id: str,
        topic: str,
        fsm_module: Any,
        nominal_period: float = 1.0,
    ) -> None:
        self.sensor_id = sensor_id
        self.topic = topic
        self.mod = fsm_module
        self.nominal_period = nominal_period

        # Instantiate fresh, isolated TFSM
        self.machine: TFSM = self.mod.build_original()
        self.state: str = self.machine.init
        self.clock: float = 0.0
        self.last_timestamp: float = 0.0
        self.sequence_number: int = 0
        self.is_connected: bool = False

    def advance_clock(self, dt: float) -> List[Tuple[float, str, str]]:
        """Advances sensor clock by dt. Emits timeout transitions if expired."""
        events = []
        remaining = dt
        while remaining > 0:
            tr = self.machine._timeout(self.state)
            if tr is not None and (self.clock + remaining >= tr.timeout - 1e-9):
                step_dt = max(tr.timeout - self.clock, 0.0)
                remaining -= step_dt
                self.clock = 0.0
                self.state = tr.dst
                events.append((step_dt, tr.out, tr.tid))
            else:
                self.clock += remaining
                remaining = 0.0
        return events

    def handle_input(self, inp: str) -> Tuple[str, str]:
        """Applies input symbol to the TFSM, transitions state, resets clock."""
        tr = self.machine._on_input(self.state, inp)
        if tr is None:
            out = "o6"  # Default TFSM invalid message error
            tid = "tx"
            # State remains unchanged on invalid input
        else:
            out = tr.out
            tid = tr.tid
            self.state = tr.dst
        self.clock = 0.0
        return out, tid


# ----------------------------------------------------------------------
# Central Unified Multi-Sensor Simulator
# ----------------------------------------------------------------------
class UnifiedSimulator:
    """Coordinates concurrent simulation of the four IoT sensors."""

    TOPICS = {
        "motion": "home/sensor/motion1",
        "temperature": "home/sensor/temperature1",
        "ultrasonic": "distance/ultrasonic1",
        "rfid": "security/rfid1",
    }

    def __init__(
        self,
        duration: float = 60.0,
        seed: int = 42,
        periods: Optional[Dict[str, float]] = None,
    ) -> None:
        self.duration = duration
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Configurable nominal reporting intervals
        self.periods = {
            "motion": 1.0,
            "temperature": 1.5,
            "ultrasonic": 0.8,
            "rfid": 2.5,
        }
        if periods:
            self.periods.update(periods)

        # Initialize the 4 sensor runtimes
        self.sensors: Dict[str, SensorRuntime] = {
            "motion": SensorRuntime("motion", self.TOPICS["motion"], motion_sensor, self.periods["motion"]),
            "temperature": SensorRuntime("temperature", self.TOPICS["temperature"], temperature_sensor, self.periods["temperature"]),
            "ultrasonic": SensorRuntime("ultrasonic", self.TOPICS["ultrasonic"], ultrasonic_sensor, self.periods["ultrasonic"]),
            "rfid": SensorRuntime("rfid", self.TOPICS["rfid"], rfid_sensor, self.periods["rfid"]),
        }

    def _generate_sensor_reading(self, sensor_id: str) -> Tuple[str, Dict[str, Any], str]:
        """Generates domain telemetry, corresponding TFSM input, and event_type."""
        if sensor_id == "motion":
            # 25% probability of motion detected
            is_motion = bool(self.rng.random() < 0.25)
            inp = "i11" if is_motion else "i5"  # i11: motion event, i5: benign health check
            event_type = "ALERT" if is_motion else "HEARTBEAT"
            payload = {
                "motion": int(is_motion),
                "confidence": round(float(self.rng.uniform(0.85, 0.99) if is_motion else 0.0), 2),
                "status": "motion_detected" if is_motion else "standby",
            }
            return inp, payload, event_type

        elif sensor_id == "temperature":
            # Normal ambient temp N(23.0, 2.0); 5% rare heat spike
            temp = float(self.rng.normal(23.0, 2.0))
            is_heat_alert = temp > 30.0
            inp = "i12" if is_heat_alert else "i11"
            event_type = "ALERT" if is_heat_alert else "PUBLISH"
            payload = {
                "temperature": round(temp, 2),
                "unit": "Celsius",
                "humidity": round(float(self.rng.uniform(40.0, 60.0)), 1),
                "alert": is_heat_alert,
            }
            return inp, payload, event_type

        elif sensor_id == "ultrasonic":
            # Distance ranging between 15cm and 250cm; <30cm triggers proximity alert
            dist = float(self.rng.uniform(15.0, 250.0))
            is_obstacle = dist < 30.0
            inp = "i12" if is_obstacle else "i11"
            event_type = "ALERT" if is_obstacle else "PUBLISH"
            payload = {
                "distance_cm": round(dist, 1),
                "echo_time_us": round(dist * 58.0, 1),
                "obstacle": is_obstacle,
            }
            return inp, payload, event_type

        elif sensor_id == "rfid":
            # Periodic badge swipe: 80% authorized card, 20% unauthorized
            is_authorized = bool(self.rng.random() < 0.80)
            inp = "i11" if is_authorized else "i12"
            event_type = "PUBLISH"
            uid = "A1B2C3D4" if is_authorized else "DEADBEEF"
            payload = {
                "card_uid": uid,
                "status": "authorized" if is_authorized else "unauthorized",
                "reader_gate": 1,
            }
            return inp, payload, event_type

        raise ValueError(f"Unknown sensor: {sensor_id}")

    def run(self) -> List[UnifiedEvent]:
        """Executes unified multi-sensor simulation across the duration.

        Returns an interleaved, chronologically sorted list of UnifiedEvent instances.
        """
        raw_events: List[UnifiedEvent] = []

        # ------------------------------------------------------------------
        # 1. Device Startup & MQTT Connect Phase for all sensors
        # ------------------------------------------------------------------
        for sensor_id, sensor in self.sensors.items():
            t_wifi = round(float(self.rng.uniform(0.05, 0.25)), 4)
            # WiFi Connected (i2 -> o2)
            out, _ = sensor.handle_input("i2")
            sensor.sequence_number += 1
            raw_events.append(
                UnifiedEvent(
                    timestamp=t_wifi,
                    sensor_id=sensor_id,
                    topic=sensor.topic,
                    event_type="CONNECT",
                    payload={"action": "wifi_connected", "status": "ok"},
                    sequence_number=sensor.sequence_number,
                    qos=1,
                    payload_size=58,
                    observed_state=sensor.state,
                    output_symbol=out,
                    inter_arrival=t_wifi,
                )
            )
            sensor.last_timestamp = t_wifi

            # MQTT Connected (i4 -> o4)
            t_mqtt = round(t_wifi + float(self.rng.uniform(0.1, 0.3)), 4)
            out, _ = sensor.handle_input("i4")
            sensor.sequence_number += 1
            raw_events.append(
                UnifiedEvent(
                    timestamp=t_mqtt,
                    sensor_id=sensor_id,
                    topic=sensor.topic,
                    event_type="CONNECT",
                    payload={"action": "mqtt_connected", "broker": "192.168.1.100:1883"},
                    sequence_number=sensor.sequence_number,
                    qos=1,
                    payload_size=64,
                    observed_state=sensor.state,
                    output_symbol=out,
                    inter_arrival=round(t_mqtt - t_wifi, 4),
                )
            )
            sensor.last_timestamp = t_mqtt
            sensor.is_connected = True

        # ------------------------------------------------------------------
        # 2. Main Interleaved Operational Simulation Loop
        # ------------------------------------------------------------------
        # Schedule next event times for each sensor
        next_times: Dict[str, float] = {}
        for sensor_id, sensor in self.sensors.items():
            jitter = float(self.rng.uniform(0.01, 0.08))
            next_times[sensor_id] = round(sensor.last_timestamp + sensor.nominal_period + jitter, 4)

        while True:
            # Find sensor with the earliest scheduled event
            sensor_id = min(next_times, key=next_times.get)
            current_time = next_times[sensor_id]

            if current_time > self.duration:
                break

            sensor = self.sensors[sensor_id]
            dt = current_time - sensor.last_timestamp

            # Check timeouts during interval dt
            timeout_events = sensor.advance_clock(dt)
            for t_offset, t_out, t_tid in timeout_events:
                timeout_ts = round(sensor.last_timestamp + t_offset, 4)
                sensor.sequence_number += 1
                raw_events.append(
                    UnifiedEvent(
                        timestamp=timeout_ts,
                        sensor_id=sensor_id,
                        topic=sensor.topic,
                        event_type="HEARTBEAT",
                        payload={"keepalive": "alive", "tid": t_tid},
                        sequence_number=sensor.sequence_number,
                        qos=0,
                        payload_size=42,
                        observed_state=sensor.state,
                        output_symbol=t_out,
                        inter_arrival=round(timeout_ts - sensor.last_timestamp, 4),
                    )
                )

            # Generate sensor telemetry input
            inp, payload, event_type = self._generate_sensor_reading(sensor_id)
            out_sym, _ = sensor.handle_input(inp)
            sensor.sequence_number += 1
            dt_actual = round(current_time - sensor.last_timestamp, 4)

            raw_events.append(
                UnifiedEvent(
                    timestamp=current_time,
                    sensor_id=sensor_id,
                    topic=sensor.topic,
                    event_type=event_type,
                    payload=payload,
                    sequence_number=sensor.sequence_number,
                    qos=1 if event_type == "ALERT" else 0,
                    payload_size=int(len(str(payload)) + 32),
                    observed_state=sensor.state,
                    output_symbol=out_sym,
                    inter_arrival=dt_actual,
                )
            )
            sensor.last_timestamp = current_time

            # Schedule next event
            jitter = float(self.rng.normal(0.0, sensor.nominal_period * 0.05))
            next_interval = max(sensor.nominal_period + jitter, 0.05)
            next_times[sensor_id] = round(current_time + next_interval, 4)

        # ------------------------------------------------------------------
        # 3. Final Ordering & Global Monotonic Timestamps
        # ------------------------------------------------------------------
        raw_events.sort(key=lambda e: (e.timestamp, e.sequence_number))

        # Recompute inter-arrival globally across the multiplexed timeline
        prev_t = 0.0
        for ev in raw_events:
            ev.inter_arrival = round(max(ev.timestamp - prev_t, 0.0), 4)
            prev_t = ev.timestamp

        return raw_events

