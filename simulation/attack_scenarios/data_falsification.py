
import json
import time
import sys
from pathlib import Path
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

# Locate mqtt_config.py inside the simulation directory.
SIMULATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATION_DIR))

from mqtt_config import BROKER_HOST, BROKER_PORT, TEMPERATURE_TOPIC

DEVICE_ID = "temp_01"
CLIENT_ID = "data_falsification_client"

# Deliberately altered readings for this controlled experiment.
FALSIFIED_TEMPERATURES = [85.0, -20.0, 95.0, 0.0, 90.0]


def main():
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=CLIENT_ID
    )

    try:
        print("Connecting to MQTT broker...")
        client.connect(BROKER_HOST, BROKER_PORT, 60)
        client.loop_start()
        time.sleep(1)

        for sequence, temperature in enumerate(
            FALSIFIED_TEMPERATURES, start=1
        ):
            payload = {
                "device_id": DEVICE_ID,
                "device_type": "temperature_sensor",
                "temperature": temperature
            }

            message = json.dumps(payload)
            result = client.publish(
                TEMPERATURE_TOPIC,
                message,
                qos=1
            )
            result.wait_for_publish(timeout=5)

            if result.is_published():
                timestamp = datetime.now(timezone.utc).isoformat()
                print(
                    f"scenario=DATA_FALSIFICATION | "
                    f"sequence={sequence} | timestamp={timestamp} | "
                    f"topic={TEMPERATURE_TOPIC} | payload={message}"
                )
            else:
                print(f"Publication {sequence} was not confirmed.")

            time.sleep(1)

        print("Data-falsification simulation finished.")

    except Exception as error:
        print(f"Simulation error: {error}")

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
