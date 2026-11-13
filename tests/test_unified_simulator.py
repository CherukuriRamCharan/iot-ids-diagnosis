"""Unit tests for Stage 2 Unified Four-Sensor IoT Event Simulator."""

from __future__ import annotations

import unittest

from simulation.unified_simulator import UnifiedEvent, UnifiedSimulator


class TestUnifiedSimulator(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = UnifiedSimulator(duration=20.0, seed=42)
        self.events = self.sim.run()

    def test_all_four_sensors_produce_events(self) -> None:
        sensor_ids = {e.sensor_id for e in self.events}
        expected_sensors = {"motion", "temperature", "ultrasonic", "rfid"}
        self.assertEqual(sensor_ids, expected_sensors)

        # Verify each sensor has multiple events
        for sid in expected_sensors:
            count = sum(1 for e in self.events if e.sensor_id == sid)
            self.assertGreater(count, 5, f"Sensor {sid} produced too few events: {count}")

    def test_sensor_states_remain_independent(self) -> None:
        # Each sensor must maintain its own state and sequence number
        seq_counters = {sid: [] for sid in ("motion", "temperature", "ultrasonic", "rfid")}
        states = {sid: set() for sid in ("motion", "temperature", "ultrasonic", "rfid")}

        for e in self.events:
            seq_counters[e.sensor_id].append(e.sequence_number)
            states[e.sensor_id].add(e.observed_state)

        for sid, seqs in seq_counters.items():
            # Sequences must be monotonically increasing per sensor
            self.assertEqual(seqs, sorted(seqs), f"Sequence not monotonic for {sid}")
            self.assertEqual(len(seqs), len(set(seqs)), f"Duplicate sequence numbers in {sid}")
            # Each sensor transitions through multiple legitimate states (e.g. s2, s5, s8, s9)
            self.assertGreaterEqual(len(states[sid]), 2, f"Sensor {sid} had stagnant states: {states[sid]}")

    def test_timestamps_are_ordered_and_monotonic(self) -> None:
        for i in range(len(self.events) - 1):
            t_curr = self.events[i].timestamp
            t_next = self.events[i + 1].timestamp
            self.assertLessEqual(t_curr, t_next, f"Timestamp out of order at idx {i}: {t_curr} > {t_next}")

    def test_reproducibility_same_seed(self) -> None:
        sim2 = UnifiedSimulator(duration=20.0, seed=42)
        events2 = sim2.run()

        self.assertEqual(len(self.events), len(events2))
        for e1, e2 in zip(self.events, events2):
            self.assertEqual(e1.timestamp, e2.timestamp)
            self.assertEqual(e1.sensor_id, e2.sensor_id)
            self.assertEqual(e1.sequence_number, e2.sequence_number)
            self.assertEqual(e1.payload, e2.payload)
            self.assertEqual(e1.output_symbol, e2.output_symbol)

    def test_variability_different_seeds(self) -> None:
        sim_diff = UnifiedSimulator(duration=20.0, seed=999)
        events_diff = sim_diff.run()

        # At least some payloads or timestamps should differ with a different seed
        timestamps1 = [e.timestamp for e in self.events[:15]]
        timestamps2 = [e.timestamp for e in events_diff[:15]]
        self.assertNotEqual(timestamps1, timestamps2)

    def test_common_message_schema_compliance(self) -> None:
        required_observable_fields = [
            "timestamp",
            "sensor_id",
            "topic",
            "event_type",
            "payload",
            "sequence_number",
            "qos",
            "payload_size",
            "observed_state",
            "output_symbol",
            "inter_arrival",
        ]
        for e in self.events:
            obs = e.to_observable_dict()
            for f in required_observable_fields:
                self.assertIn(f, obs, f"Missing observable field: {f}")

            # Topics must match formal sensor topics
            self.assertEqual(e.topic, self.sim.TOPICS[e.sensor_id])
            self.assertGreater(e.payload_size, 0)
            self.assertGreaterEqual(e.inter_arrival, 0.0)

    def test_ground_truth_metadata_separation(self) -> None:
        quarantined_keys = ["is_attack", "attack_type", "attack_target", "fault_location"]
        for e in self.events:
            obs = e.to_observable_dict()
            for qk in quarantined_keys:
                self.assertNotIn(qk, obs, f"Leakage! Quarantined key {qk} found in observable dict")

            gt = e.to_ground_truth_dict()
            for qk in quarantined_keys:
                self.assertIn(qk, gt, f"Ground-truth key {qk} missing from ground truth dict")

    def test_sensor_startup_phase(self) -> None:
        # Check that the first two events for each sensor are CONNECT (wifi then mqtt)
        for sid in ("motion", "temperature", "ultrasonic", "rfid"):
            sensor_events = [e for e in self.events if e.sensor_id == sid]
            self.assertGreaterEqual(len(sensor_events), 2)
            self.assertEqual(sensor_events[0].event_type, "CONNECT")
            self.assertEqual(sensor_events[0].output_symbol, "o2")  # WiFi ok
            self.assertEqual(sensor_events[1].event_type, "CONNECT")
            self.assertEqual(sensor_events[1].output_symbol, "o4")  # MQTT ok


if __name__ == "__main__":
    unittest.main()

