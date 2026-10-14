
import json
import time
import sys
from pathlib import Path
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

# Locate the existing MQTT configuration.
SIMULATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATION_DIR))

from mqtt_config import BROKER_HOST, BROKER_PORT, TEMPERATURE_TOPIC

DEVICE_ID = "temp_01"
CLIENT_ID = "battery_draining_test_client"

# Bounded experiment: 100 messages at a short interval.
MESSAGE_COUNT = 100
PUBLISH_INTERVAL = 0.05  # 50 milliseconds


def main():
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=CLIENT_ID
    )

    successful_publications = 0
    start_time = time.perf_counter()

    try:
        print(f"Connecting to {BROKER_HOST}:{BROKER_PORT}...")
        client.connect(BROKER_HOST, BROKER_PORT, 60)
        client.loop_start()

        # Allow the connection callback and network loop to initialize.
        time.sleep(1)

        start_time = time.perf_counter()

        for sequence in range(1, MESSAGE_COUNT + 1):
            payload = {
                "device_id": DEVICE_ID,
                "device_type": "temperature_sensor",
                "sequence": sequence,
                "temperature": 27.5,
                "scenario": "high_frequency_communication"
            }

            message = json.dumps(payload)

            result = client.publish(
                TEMPERATURE_TOPIC,
                message,
                qos=1
            )

            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                successful_publications += 1
            else:
                print(
                    f"Local publish request failed at "
                    f"sequence {sequence}: "
                    f"{mqtt.error_string(result.rc)}"
                )

            time.sleep(PUBLISH_INTERVAL)

        # Wait for queued QoS 1 publications to complete.
        client.loop_stop()

        elapsed = time.perf_counter() - start_time
        rate = (
            successful_publications / elapsed
            if elapsed > 0
            else 0
        )

        print("\n--- High-Frequency Communication Summary ---")
        print(f"Scenario: battery-draining-like")
        print(f"Topic: {TEMPERATURE_TOPIC}")
        print(f"Requested messages: {MESSAGE_COUNT}")
        print(
            f"Accepted publish requests: "
            f"{successful_publications}"
        )
        print(f"Elapsed time: {elapsed:.2f} seconds")
        print(f"Request rate: {rate:.2f} messages/second")
        print(
            "Note: accepted requests do not by themselves "
            "prove broker receipt or battery depletion."
        )

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

    except Exception as error:
        print(f"Simulation error: {error}")

    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
