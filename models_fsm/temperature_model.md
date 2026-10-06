# Temperature Sensor Formal Model

## 1. Device

- Device ID: `temp_01`
- Device Type: Temperature Sensor
- MQTT Topic: `iot/temperature`
- MQTT Broker Port: `1883`
- Publishing Interval: `2 seconds`

## 2. Purpose

This model represents the behavior of the simulated temperature
sensor used in the IoT MQTT environment.

The model represents normal device behavior using states,
events, transitions, timing behavior and variables.

An abnormal transition is also included as a placeholder for
the attack-scenario generation stage.

## 3. States

| State | Description |
|---|---|
| INIT | Device has not established an MQTT connection |
| CONNECTED | Device is connected to the MQTT broker |
| GENERATE_DATA | Device generates a temperature reading |
| PUBLISH_DATA | Device publishes the generated reading |
| WAIT_TIMER | Device waits for the next publishing interval |
| DISCONNECTED | Device is disconnected |
| ANOMALOUS | Unexpected communication behavior is observed |

## 4. Variables

| Variable | Description |
|---|---|
| device_id | Identifies the IoT device |
| mqtt_topic | MQTT topic used by the device |
| publish_interval | Time interval between publications |
| temperature | Generated temperature value |
| state | Current state of the formal model |

## 5. Events / Inputs

| Event | Description |
|---|---|
| connect | MQTT connection is established |
| timer_expired | Publishing interval has elapsed |
| data_generated | Temperature value has been generated |
| publish_success | MQTT message is successfully published |
| unexpected_message | Unexpected communication is observed |
| disconnect | Device disconnects from the broker |

## 6. Normal Transitions

| Current State | Event | Action | Next State |
|---|---|---|---|
| INIT | connect | Establish MQTT connection | CONNECTED |
| CONNECTED | timer_expired | Start data generation | GENERATE_DATA |
| GENERATE_DATA | data_generated | Store temperature value | PUBLISH_DATA |
| PUBLISH_DATA | publish_success | Publish MQTT message | WAIT_TIMER |
| WAIT_TIMER | timer_expired | Start next data generation | GENERATE_DATA |
| CONNECTED | disconnect | Close connection | DISCONNECTED |

## 7. Abnormal Transition

| Current State | Event | Action | Next State |
|---|---|---|---|
| PUBLISH_DATA | unexpected_message | Record abnormal communication | ANOMALOUS |

The abnormal state is currently a placeholder for the controlled
attack scenarios that will be generated during the mutation and
attack-generation stage.

## 8. Timing Behavior

The temperature sensor publishes approximately every 2 seconds.

The timing sequence is:

INIT
→ CONNECTED
→ GENERATE_DATA
→ PUBLISH_DATA
→ WAIT_TIMER
→ GENERATE_DATA
→ PUBLISH_DATA
→ ...

The timing behavior is important because the project uses
timed formal modeling to represent IoT device behavior.

## 9. Mapping to the Python Simulation

| Python Simulation | Formal Model |
|---|---|
| `client.connect()` | INIT → CONNECTED |
| Temperature generation | GENERATE_DATA |
| `client.publish()` | PUBLISH_DATA |
| `time.sleep(2)` | WAIT_TIMER |
| `client.disconnect()` | DISCONNECTED |

## 10. Model Validation

The formal model was executed using:

`models_fsm/temperature_efsm.py`

The execution successfully demonstrated:

1. MQTT connection
2. Temperature data generation
3. MQTT publication
4. Waiting for the next interval
5. Repeated temperature generation and publication
6. Transition to the abnormal state when an unexpected event occurs

## 11. Relation to Attack Generation

The formal model provides a structured representation of the
expected device behavior.

In the next stage, mutation and attack scenarios will be developed
from the modeled behavior.

The current model does not implement the actual attack scenarios.
Those will be handled separately during the mutation and attack
generation stage.

## 12. Formal Model Diagram

```text
                         ┌─────────────┐
                         │    INIT     │
                         └──────┬──────┘
                                │ connect
                                ▼
                       ┌────────────────┐
                       │   CONNECTED    │
                       └───────┬────────┘
                               │ timer_expired
                               ▼
                       ┌────────────────┐
                       │ GENERATE_DATA  │
                       └───────┬────────┘
                               │ data_generated
                               ▼
                       ┌────────────────┐
                       │  PUBLISH_DATA  │
                       └───────┬────────┘
                               │ publish_success
                               ▼
                       ┌────────────────┐
                       │   WAIT_TIMER   │
                       └───────┬────────┘
                               │ timer_expired
                               │
                               └──────────────►
                                      GENERATE_DATA


                 PUBLISH_DATA
                       │
                       │ unexpected_message
                       ▼
                ┌──────────────┐
                │   ANOMALOUS  │
                └──────────────┘


        CONNECTED / WAIT_TIMER / GENERATE_DATA /
        PUBLISH_DATA / ANOMALOUS
                       │
                       │ disconnect
                       ▼
                ┌──────────────┐
                │ DISCONNECTED │
                └──────────────┘