"""
Motion Sensor Extended Finite-State Machine (EFSM).

Models the expected behaviour of the simulated PIR/motion sensor
publishing binary (0/1) motion readings on iot/motion via MQTT.

States, transitions, variables, and attack detectors mirror the
temperature sensor EFSM in temperature_efsm.py; the payload
semantics differ (binary motion flag instead of numeric temperature).

Paper reference: Kaddoura et al. (2026), §3 — four device types are
modelled; this is the motion-sensor model.
"""

from dataclasses import dataclass, field
from collections import deque
from time import monotonic


# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------
INIT         = "INIT"
CONNECTED    = "CONNECTED"
WAIT_TIMER   = "WAIT_TIMER"
SENSE        = "SENSE"
PUBLISH_DATA = "PUBLISH_DATA"
DISCONNECTED = "DISCONNECTED"
ANOMALOUS    = "ANOMALOUS"


@dataclass
class MotionEFSM:
    """
    Motion sensor EFSM with four rule-based detectors aligned with
    the paper's attack mutant catalogue.

    Expected normal behaviour:
      INIT -> CONNECTED -> SENSE -> PUBLISH_DATA -> WAIT_TIMER
           -> SENSE -> PUBLISH_DATA -> ...

    Valid motion values: 0 (no motion) or 1 (motion detected).
    Publish interval: approximately every 2 seconds.
    """

    device_id: str = "motion_01"
    mqtt_topic: str = "iot/motion"
    publish_interval: float = 2.0
    max_messages_per_window: int = 5
    window_seconds: float = 10.0

    motion: int | None = None
    previous_motion: int | None = None
    state: str = INIT
    detected_attacks: list[str] = field(default_factory=list)
    publish_times: deque = field(default_factory=deque)

    # --- Normal lifecycle transitions ---

    def connect(self):
        if self.state == INIT:
            self.state = CONNECTED
            return "MQTT connection established"
        return "Invalid transition: device is not in INIT state"

    def timer_expired(self):
        if self.state in (CONNECTED, WAIT_TIMER):
            self.state = SENSE
            return "Timer expired: sensing motion"
        return "Invalid transition: timer event not allowed in current state"

    def motion_sensed(self, value: int):
        if self.state != SENSE:
            return "Invalid transition: cannot sense outside SENSE state"
        if value not in (0, 1):
            return self._flag(f"Invalid motion value: {value!r} (expected 0 or 1)")
        self.motion = value
        self.state = PUBLISH_DATA
        return f"Motion value sensed: {value}"

    def publish_success(self):
        if self.state != PUBLISH_DATA:
            return "Invalid transition: publish not allowed in current state"
        self.state = WAIT_TIMER
        return f"Published to {self.mqtt_topic}"

    def disconnect(self):
        if self.state in (CONNECTED, WAIT_TIMER, SENSE, PUBLISH_DATA, ANOMALOUS):
            self.state = DISCONNECTED
            return "Device disconnected"
        return "Invalid transition: device cannot disconnect from current state"

    def unexpected_message(self):
        if self.state == PUBLISH_DATA:
            return self._flag("Unexpected communication during PUBLISH_DATA")
        return "Invalid: unexpected_message not applicable in current state"

    # --- Attack detectors ---

    def _flag(self, attack: str) -> str:
        if attack not in self.detected_attacks:
            self.detected_attacks.append(attack)
        self.state = ANOMALOUS
        return f"DETECTED: {attack}"

    def detect_replay(self, payload: int):
        """
        Flag when consecutive motion values are identical.

        Note: consecutive identical readings (e.g., sustained no-motion)
        can be legitimate.  This is a contextual signal, not a definitive
        attack indicator.  Distinguish from sensor inactivity by examining
        timing intervals.
        """
        if self.previous_motion is not None and payload == self.previous_motion:
            # Record as evidence only; do not enter ANOMALOUS for a single repeat.
            evidence = f"Replay-like: repeated motion value {payload}"
            if evidence not in self.detected_attacks:
                self.detected_attacks.append(evidence)
            return f"EVIDENCE: {evidence}"
        self.previous_motion = payload
        return "No replay-like duplicate detected"

    def detect_invalid_value(self, value) -> str:
        """
        Flag any motion value that is not binary (0 or 1).
        In the paper's context this maps to 'data falsification'.
        """
        try:
            v = int(value)
        except (TypeError, ValueError):
            return self._flag(f"Data falsification: non-numeric motion value {value!r}")
        if v not in (0, 1):
            return self._flag(
                f"Data falsification: motion value {v} outside valid range {{0, 1}}"
            )
        return "Motion value is valid (0 or 1)"

    def detect_message_manipulation(
        self, original_value: int, forwarded_value: int
    ) -> str:
        """
        Detect when a forwarded payload differs from the original.
        Application-level only (not network-path MITM).
        """
        if original_value != forwarded_value:
            return self._flag(
                f"Message manipulation: original {original_value} "
                f"forwarded as {forwarded_value}"
            )
        return "No payload modification detected"

    def detect_excessive_communication(self, timestamp=None) -> str:
        """
        Detect a publish rate exceeding the configured threshold within
        the rolling window.  Used to represent battery-draining behaviour.
        """
        now = monotonic() if timestamp is None else timestamp
        self.publish_times.append(now)

        while self.publish_times and now - self.publish_times[0] > self.window_seconds:
            self.publish_times.popleft()

        if len(self.publish_times) > self.max_messages_per_window:
            return self._flag(
                f"Excessive communication: "
                f"{len(self.publish_times)} publishes in {self.window_seconds}s "
                f"(threshold: {self.max_messages_per_window})"
            )
        return "Communication rate within configured limit"


# ---------------------------------------------------------------------------
# Built-in self-tests
# ---------------------------------------------------------------------------

def run_detection_tests():
    print("=== Motion EFSM — Detection Tests ===\n")

    print("--- 1. Normal lifecycle ---")
    m = MotionEFSM()
    print(m.connect())
    print(m.timer_expired())
    print(m.motion_sensed(1))
    print(m.publish_success())
    print(m.timer_expired())
    print(m.motion_sensed(0))
    print(m.publish_success())
    print("Final state:", m.state, "\n")

    print("--- 2. Replay-like (repeated value) ---")
    m = MotionEFSM()
    print(m.detect_replay(0))
    print(m.detect_replay(0))   # second identical value -> evidence
    print("Detected attacks:", m.detected_attacks, "\n")

    print("--- 3. Data falsification (invalid value) ---")
    m = MotionEFSM()
    print(m.detect_invalid_value(1))   # valid
    print(m.detect_invalid_value(2))   # invalid
    print("State:", m.state, "\n")

    print("--- 4. Message manipulation ---")
    m = MotionEFSM()
    print(m.detect_message_manipulation(original_value=1, forwarded_value=0))
    print("State:", m.state, "\n")

    print("--- 5. Excessive communication (battery draining) ---")
    m = MotionEFSM(max_messages_per_window=3, window_seconds=10)
    for t in [0.5, 1.0, 1.5, 2.0]:
        print(m.detect_excessive_communication(timestamp=t))
    print("State:", m.state)


if __name__ == "__main__":
    run_detection_tests()
