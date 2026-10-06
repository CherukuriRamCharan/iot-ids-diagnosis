"""
Formal model for the simulated temperature sensor.

Device:
    temp_01

MQTT Topic:
    iot/temperature

Purpose:
    Represent the normal timed behavior of the temperature sensor
    and provide an abnormal transition for later attack generation.
"""

from dataclasses import dataclass


# -----------------------------
# States
# -----------------------------

INIT = "INIT"
CONNECTED = "CONNECTED"
WAIT_TIMER = "WAIT_TIMER"
GENERATE_DATA = "GENERATE_DATA"
PUBLISH_DATA = "PUBLISH_DATA"
DISCONNECTED = "DISCONNECTED"
ANOMALOUS = "ANOMALOUS"


@dataclass
class TemperatureEFSM:
    """Simple EFSM representation of the temperature sensor."""

    device_id: str = "temp_01"
    mqtt_topic: str = "iot/temperature"
    publish_interval: int = 2
    temperature: float | None = None
    state: str = INIT

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
        if self.state == GENERATE_DATA:
            self.temperature = temperature
            self.state = PUBLISH_DATA
            return f"Temperature generated: {temperature}"

        return "Invalid transition: data cannot be generated"

    def publish_success(self):
        if self.state == PUBLISH_DATA:
            self.state = WAIT_TIMER
            return f"Published to {self.mqtt_topic}"

        return "Invalid transition: publish not allowed"

    def unexpected_message(self):
        if self.state == PUBLISH_DATA:
            self.state = ANOMALOUS
            return "Abnormal communication detected"

        return "Invalid abnormal transition"

    def disconnect(self):
        if self.state in (
            CONNECTED,
            WAIT_TIMER,
            GENERATE_DATA,
            PUBLISH_DATA,
            ANOMALOUS,
        ):
            self.state = DISCONNECTED
            return "Device disconnected"

        return "Invalid transition: device cannot disconnect"


def run_normal_scenario():
    """Demonstrate the normal temperature-sensor behavior."""

    model = TemperatureEFSM()

    print("Initial state:", model.state)

    print(model.connect())
    print("Current state:", model.state)

    print(model.timer_expired())
    print("Current state:", model.state)

    print(model.data_generated(27.57))
    print("Current state:", model.state)

    print(model.publish_success())
    print("Current state:", model.state)

    print(model.timer_expired())
    print("Current state:", model.state)

    print(model.data_generated(28.12))
    print("Current state:", model.state)

    print(model.publish_success())
    print("Current state:", model.state)


def run_abnormal_scenario():
    """Demonstrate the abnormal transition placeholder."""

    model = TemperatureEFSM()

    model.connect()
    model.timer_expired()
    model.data_generated(27.57)

    print("\nBefore abnormal event:", model.state)

    print(model.unexpected_message())

    print("After abnormal event:", model.state)


if __name__ == "__main__":
    print("=== Normal Temperature Sensor Scenario ===")
    run_normal_scenario()

    print("\n=== Abnormal Transition Scenario ===")
    run_abnormal_scenario()