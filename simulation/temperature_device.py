import json
import time
import random

import paho.mqtt.client as mqtt

from mqtt_config import BROKER_HOST, BROKER_PORT, TEMPERATURE_TOPIC


DEVICE_ID = "temp_01"
CLIENT_ID = "temperature_device_01"


client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id=CLIENT_ID
)

client.connect(BROKER_HOST, BROKER_PORT, 60)

print("Temperature device connected to MQTT broker.")

try:
    while True:

        temperature = round(random.uniform(24.0, 30.0), 2)

        payload = {
            "device_id": DEVICE_ID,
            "device_type": "temperature_sensor",
            "temperature": temperature
        }

        message = json.dumps(payload)

        client.publish(TEMPERATURE_TOPIC, message)

        print(
            f"Published to {TEMPERATURE_TOPIC}: {message}"
        )

        time.sleep(2)

except KeyboardInterrupt:
    print("\nTemperature device stopped.")

finally:
    client.disconnect()