
from dataclasses import dataclass, field
from collections import deque
from time import monotonic

# States
INIT = "INIT"
CONNECTED = "CONNECTED"
WAIT_TIMER = "WAIT_TIMER"
GENERATE_DATA = "GENERATE_DATA"
PUBLISH_DATA = "PUBLISH_DATA"
DISCONNECTED = "DISCONNECTED"
ANOMALOUS = "ANOMALOUS"


@dataclass
class TemperatureEFSM:
    """Temperature sensor EFSM with four rule-based detectors."""

    device_id: str = "temp_01"
    mqtt_topic: str = "iot/temperature"
    publish_interval: float = 2.0
    min_temperature: float = -10.0
    max_temperature: float = 60.0
    max_messages_per_window: int = 5
    window_seconds: float = 10.0

    temperature: float | None = None
    previous_temperature: float | None = None
    state: str = INIT
    detected_attacks: list[str] = field(default_factory=list)
    publish_times: deque = field(default_factory=deque)

    def connect(self):
        if self.state == INIT:
            self.state = CONNECTED
            return "MQTT connection established"
        return "Invalid transition: device is not in INIT state"

    def timer_expired(self):
        if self.state in (CONNECTED, WAIT_TIMER):
            self.state = GENERATE_DATA
            return "Timer expired: generating temperature data"
        return "Invalid transition: timer event not allowed"

    def data_generated(self, temperature: float):
        if self.state != GENERATE_DATA:
            return "Invalid transition: data cannot be generated"

        self.temperature = temperature
        self.state = PUBLISH_DATA
        return f"Temperature generated: {temperature}"

    def _flag(self, attack: str):
        if attack not in self.detected_attacks:
            self.detected_attacks.append(attack)
        self.state = ANOMALOUS
        return f"DETECTED: {attack}"

    def detect_replay(self, payload, timestamp=None, sequence=None):
        """Detect duplicate payloads when called on received messages."""
        if self.previous_temperature is not None and payload == self.previous_temperature:
            return self._flag("Replay-like: repeated payload")

        self.previous_temperature = payload
        return "No replay-like duplicate detected"

    def detect_data_falsification(self, temperature: float):
        if not self.min_temperature <= temperature <= self.max_temperature:
            return self._flag("Data falsification: out-of-range temperature")
        return "No out-of-range temperature detected"

    def detect_message_manipulation(self, original_payload, forwarded_payload):
        if original_payload != forwarded_payload:
            return self._flag("Message manipulation: payload changed")
        return "No payload modification detected"

    def detect_excessive_communication(self, timestamp=None):
        """Detect more than the configured message limit per time window."""
        now = monotonic() if timestamp is None else timestamp
        self.publish_times.append(now)

        while self.publish_times and now - self.publish_times[0] > self.window_seconds:
            self.publish_times.popleft()

        if len(self.publish_times) > self.max_messages_per_window:
            return self._flag("Excessive communication: publish rate too high")

        return "Communication rate within configured limit"

    def publish_success(self):
        if self.state != PUBLISH_DATA:
            return "Invalid transition: publish not allowed"

        self.state = WAIT_TIMER
        return f"Published to {self.mqtt_topic}"

    def unexpected_message(self):
        if self.state == PUBLISH_DATA:
            return self._flag("Unexpected communication")
        return "Invalid abnormal transition"

    def disconnect(self):
        if self.state in (
            CONNECTED, WAIT_TIMER, GENERATE_DATA,
            PUBLISH_DATA, ANOMALOUS
        ):
            self.state = DISCONNECTED
            return "Device disconnected"
        return "Invalid transition: device cannot disconnect"


def run_detection_tests():
    print("=== 1. Replay-like Detection ===")
    model = TemperatureEFSM()
    print(model.detect_replay(27.57))
    print(model.detect_replay(27.57))
    print("State:", model.state)

    print("\n=== 2. Data Falsification Detection ===")
    model = TemperatureEFSM()
    print(model.detect_data_falsification(27.0))
    print(model.detect_data_falsification(95.0))
    print("State:", model.state)

    print("\n=== 3. Message Manipulation Detection ===")
    model = TemperatureEFSM()
    print(model.detect_message_manipulation(
        original_payload=27.0,
        forwarded_payload=42.0
    ))
    print("State:", model.state)

    print("\n=== 4. Excessive Communication Detection ===")
    model = TemperatureEFSM(max_messages_per_window=3, window_seconds=10)
    for timestamp in [1.0, 1.5, 2.0, 2.5]:
        print(model.detect_excessive_communication(timestamp))
    print("State:", model.state)


if __name__ == "__main__":
    run_detection_tests()
