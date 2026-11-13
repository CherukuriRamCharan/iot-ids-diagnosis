# Motion Sensor Formal Model

## 1. Device

- Device ID: `motion_01`
- Device Type: PIR / Motion Sensor (binary)
- MQTT Topic: `iot/motion`
- MQTT Broker Port: `1883`
- Publishing Interval: `2 seconds`
- Valid Payload Values: `0` (no motion) or `1` (motion detected)

## 2. Purpose

This model represents the expected behaviour of the simulated PIR motion
sensor in the IoT MQTT environment.  It is the second device model in the
project, complementing the temperature sensor EFSM.

Paper reference: Kaddoura et al. (2026), §3 — four device types are modelled
(temperature sensor, motion sensor, ultrasonic motion sensor, RFID reader).
This is the motion-sensor model.

## 3. States

| State | Description |
|---|---|
| INIT | Device has not established an MQTT connection |
| CONNECTED | Device is connected to the MQTT broker |
| SENSE | Device samples the PIR sensor for a motion reading |
| PUBLISH_DATA | Device publishes the binary reading |
| WAIT_TIMER | Device waits for the next publishing interval |
| DISCONNECTED | Device is disconnected from the broker |
| ANOMALOUS | An unexpected or invalid event has been observed |

## 4. Variables

| Variable | Type | Description |
|---|---|---|
| device_id | string | Identifies the IoT device |
| mqtt_topic | string | MQTT topic used by the device (`iot/motion`) |
| publish_interval | float | Nominal interval between publications (seconds) |
| motion | int ∈ {0,1} | Most recently sensed motion value |
| previous_motion | int ∈ {0,1} | Previous sensed value (replay detection) |
| state | string | Current EFSM state |
| publish_times | deque | Ring buffer of recent publish timestamps |
| max_messages_per_window | int | Rate threshold (default 5 per 10 s) |
| window_seconds | float | Rate-detection window (default 10 s) |

## 5. Events / Inputs

| Event | Description |
|---|---|
| connect | MQTT connection established |
| timer_expired | Publishing interval has elapsed |
| motion_sensed(v) | PIR sensor returns value v ∈ {0,1} |
| publish_success | MQTT PUBLISH acknowledged |
| unexpected_message | Out-of-sequence communication observed |
| disconnect | Device disconnects from the broker |

## 6. Normal Transitions

| Current State | Event | Guard | Action | Next State |
|---|---|---|---|---|
| INIT | connect | — | Establish MQTT connection | CONNECTED |
| CONNECTED | timer_expired | — | Start sensing | SENSE |
| SENSE | motion_sensed(v) | v ∈ {0,1} | Store motion value | PUBLISH_DATA |
| PUBLISH_DATA | publish_success | — | Publish to iot/motion | WAIT_TIMER |
| WAIT_TIMER | timer_expired | — | Start next sense cycle | SENSE |
| Any active state | disconnect | — | Close connection | DISCONNECTED |

## 7. Abnormal Transitions

| Trigger | Condition | Action | Next State |
|---|---|---|---|
| motion_sensed(v) | v ∉ {0,1} | Flag data falsification | ANOMALOUS |
| detect_message_manipulation | original ≠ forwarded | Flag manipulation | ANOMALOUS |
| detect_excessive_communication | count > threshold in window | Flag battery draining | ANOMALOUS |
| unexpected_message in PUBLISH_DATA | — | Flag unexpected comms | ANOMALOUS |

Replay-like behaviour (repeated consecutive identical values) is recorded as
**contextual evidence only** and does not immediately transition to ANOMALOUS,
because legitimate no-motion intervals produce sustained `0` readings.

## 8. Timing Behaviour

The motion sensor publishes approximately every 2 seconds (same as
temperature sensor).  The timing sequence is:

```
INIT → CONNECTED → SENSE → PUBLISH_DATA → WAIT_TIMER → SENSE → ...
```

The `publish_interval` is stored as a model variable but not enforced by the
EFSM implementation itself (the simulation loop enforces it via `time.sleep`).

## 9. Attack Mutants

| Mutant | Observable Symptom | EFSM Signal |
|---|---|---|
| Replay-like | Repeated identical motion values published | `repeated_motion_value` (evidence) |
| Data falsification | Value outside {0,1} (e.g. 2, -1, "abc") | `data_falsification` (ANOMALOUS) |
| Message manipulation | Forwarded value differs from original | `message_manipulation` (ANOMALOUS) |
| Battery draining | > threshold publishes in rolling window | `excessive_communication` (ANOMALOUS) |

## 10. Formal Model Diagram

```text
                     ┌──────────┐
                     │   INIT   │
                     └────┬─────┘
                          │ connect
                          ▼
                    ┌───────────┐
                    │ CONNECTED │
                    └─────┬─────┘
                          │ timer_expired
                          ▼
                    ┌───────────┐
                    │   SENSE   │◄──────────────────┐
                    └─────┬─────┘                   │
                          │ motion_sensed(v∈{0,1})  │
                          ▼                         │
                    ┌─────────────┐                 │
                    │ PUBLISH_DATA│                 │
                    └─────┬───────┘                 │
                          │ publish_success         │
                          ▼                         │
                    ┌────────────┐  timer_expired   │
                    │ WAIT_TIMER │──────────────────┘
                    └────────────┘


           On any abnormal event:
                    ┌───────────┐
                    │ ANOMALOUS │
                    └───────────┘

           On disconnect from any active state:
                    ┌──────────────┐
                    │ DISCONNECTED │
                    └──────────────┘
```

## 11. Mapping to Python Simulation

| Python Simulation | Formal Model |
|---|---|
| `client.connect()` | INIT → CONNECTED |
| `time.sleep(2)` | WAIT_TIMER → SENSE |
| `random.randint(0, 1)` | SENSE state — generates motion value |
| `client.publish()` | PUBLISH_DATA → WAIT_TIMER |
| `client.disconnect()` | → DISCONNECTED |

## 12. Limitations

- The simulation generates motion values using `random.randint(0,1)`,
  not a real PIR sensor.
- Replay detection for binary motion values is inherently ambiguous:
  legitimate sustained no-motion produces many `0` readings.
- No simulation has been developed specifically for the motion sensor's
  abnormal scenarios yet (attack captures exist only for the temperature sensor).
- The paper also models ultrasonic motion sensor and RFID reader;
  those are not yet implemented in this project (scope limit for final year).
