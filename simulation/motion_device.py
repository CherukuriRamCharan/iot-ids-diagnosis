import json
import time
import random

import paho.mqtt.client as mqtt

from mqtt_config import BROKER_HOST, BROKER_PORT, MOTION_TOPIC


DEVICE_ID = "motion_01"
CLIENT_ID = "motion_device_01"


client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id=CLIENT_ID
)

client.connect(BROKER_HOST, BROKER_PORT, 60)

print("Motion device connected to MQTT broker.")

try:
    while True:

        motion = random.choice([0, 1])

        payload = {
            "device_id": DEVICE_ID,
            "device_type": "motion_sensor",
            "motion": motion
        }

        message = json.dumps(payload)

        client.publish(MOTION_TOPIC, message)

        print(
            f"Published to {MOTION_TOPIC}: {message}"
        )

        time.sleep(2)

except KeyboardInterrupt:
    print("\nMotion device stopped.")

finally:
    client.disconnect()