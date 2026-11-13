
import json
import time
import sys
from pathlib import Path
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

# Locate the existing MQTT configuration.
SIMULATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATION_DIR))

from mqtt_config import BROKER_HOST, BROKER_PORT

DEVICE_ID = "temp_01"

INPUT_TOPIC = "iot/test/temperature/original"
OUTPUT_TOPIC = "iot/test/temperature/forwarded"

PUBLISHER_CLIENT_ID = "controlled_manipulation_publisher"
SUBSCRIBER_CLIENT_ID = "controlled_manipulation_subscriber"

TEST_READINGS = [26.5, 27.0, 27.4, 28.1, 27.8]

received_count = 0
forwarded_count = 0


def on_output_message(client, userdata, message):
    """Verify that a modified message reaches the output topic."""
    global received_count

    try:
        payload = json.loads(message.payload.decode("utf-8"))
        received_count += 1

        print(
            f"VERIFIED_OUTPUT | count={received_count} | "
            f"topic={message.topic} | "
            f"temperature={payload.get('temperature')} | "
            f"device_id={payload.get('device_id')}"
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        print(f"Could not decode output message: {error}")


def on_input_message(client, userdata, message):
    """Modify a controlled test message and forward it asynchronously."""
    global forwarded_count

    try:
        original = json.loads(message.payload.decode("utf-8"))

        if original.get("device_id") != DEVICE_ID:
            print("Ignored message from an unexpected device.")
            return

        original_value = float(original["temperature"])
        modified_value = round(original_value + 15.0, 2)

        modified = {
            **original,
            "temperature": modified_value,
            "simulation_note": "controlled_message_manipulation"
        }

        # Do not wait for publication inside this network callback.
        result = client.publish(
            OUTPUT_TOPIC,
            json.dumps(modified),
            qos=1
        )

        timestamp = datetime.now(timezone.utc).isoformat()

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            forwarded_count += 1
            print(
                f"MANIPULATED | timestamp={timestamp} | "
                f"original={original_value} | "
                f"modified={modified_value} | "
                f"status=queued"
            )
        else:
            print(
                f"Forwarding request failed: "
                f"{mqtt.error_string(result.rc)}"
            )

    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
        KeyError
    ) as error:
        print(f"Invalid input message: {error}")


def main():
    subscriber = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=SUBSCRIBER_CLIENT_ID
    )
    subscriber.on_message = on_output_message

    publisher = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=PUBLISHER_CLIENT_ID
    )
    publisher.on_message = on_input_message

    try:
        # Connect the output verifier first.
        subscriber.connect(BROKER_HOST, BROKER_PORT, 60)
        subscriber.subscribe(OUTPUT_TOPIC, qos=1)
        subscriber.loop_start()

        # Connect the controlled manipulation component.
        publisher.connect(BROKER_HOST, BROKER_PORT, 60)
        publisher.subscribe(INPUT_TOPIC, qos=1)
        publisher.loop_start()

        # Give both subscriptions time to become active.
        time.sleep(2)

        for sequence, temperature in enumerate(
            TEST_READINGS, start=1
        ):
            payload = {
                "device_id": DEVICE_ID,
                "device_type": "temperature_sensor",
                "sequence": sequence,
                "temperature": temperature
            }

            info = subscriber.publish(
                INPUT_TOPIC,
                json.dumps(payload),
                qos=1
            )
            info.wait_for_publish(timeout=5)

            if info.is_published():
                print(
                    f"INPUT_PUBLISHED | sequence={sequence} | "
                    f"temperature={temperature}"
                )
            else:
                print(
                    f"Input publication {sequence} "
                    f"was not confirmed."
                )

            time.sleep(1)

        # Allow callbacks and output verification to finish.
        time.sleep(3)

        print("\n--- Simulation Summary ---")
        print(f"Modified messages queued: {forwarded_count}")
        print(f"Modified messages received: {received_count}")

        if received_count == len(TEST_READINGS):
            print("Result: All modified messages were verified.")
        else:
            print(
                "Result: Verification incomplete. "
                "Check the output counts before recording success."
            )

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

    except Exception as error:
        print(f"Simulation error: {error}")

    finally:
        publisher.loop_stop()
        subscriber.loop_stop()
        publisher.disconnect()
        subscriber.disconnect()


if __name__ == "__main__":
    main()
