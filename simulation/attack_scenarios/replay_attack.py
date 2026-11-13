
import json
import time
import sys
from pathlib import Path
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

# Allow Python to find mqtt_config.py in the parent simulation folder.
SIMULATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATION_DIR))

from mqtt_config import BROKER_HOST, BROKER_PORT, TEMPERATURE_TOPIC

DEVICE_ID = "temp_01"
CLIENT_ID = "replay_attack_client"

# Example of a previously recorded temperature payload.
RECORDED_PAYLOAD = {
    "device_id": DEVICE_ID,
    "device_type": "temperature_sensor",
    "temperature": 27.57
}


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"Connected to MQTT broker at {BROKER_HOST}:{BROKER_PORT}")
    else:
        print(f"MQTT connection failed: {reason_code}")


def main():
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=CLIENT_ID
    )
    client.on_connect = on_connect

    try:
        print("Connecting to MQTT broker...")
        client.connect(BROKER_HOST, BROKER_PORT, 60)
        client.loop_start()

        # Wait briefly for the connection callback.
        time.sleep(1)

        for sequence in range(1, 6):
            message = json.dumps(RECORDED_PAYLOAD)

            result = client.publish(
                TEMPERATURE_TOPIC,
                message,
                qos=1
            )
            result.wait_for_publish(timeout=5)

            if result.is_published():
                timestamp = datetime.now(timezone.utc).isoformat()
                print(
                    f"scenario=REPLAY | sequence={sequence} | "
                    f"timestamp={timestamp} | topic={TEMPERATURE_TOPIC} | "
                    f"payload={message}"
                )
            else:
                print(f"Publication {sequence} was not confirmed.")

            time.sleep(1)

        print("Replay simulation finished.")

    except Exception as error:
        print(f"Replay simulation error: {error}")

    except KeyboardInterrupt:
        print("\nReplay simulation stopped by user.")

    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
