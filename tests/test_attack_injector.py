"""Unit tests for Stage 2 Sensor-Aware Attack Injector.

Verifies:
1. Denial of Service (DoS)
2. Replay Attacks
3. Man-in-the-Middle (MITM)
4. Data Falsification (DFA)
5. Message Delay
6. RFID Identity Spoofing
7. Multi-attack scenario interleaving & global timestamp monotonicity
8. Strict feature/label anti-leakage separation
"""
from __future__ import annotations

import unittest

from simulation.attack_injector import AttackInjector, AttackSpec
from simulation.unified_simulator import UnifiedSimulator


class TestAttackInjector(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = UnifiedSimulator(duration=30.0, seed=42)
        self.raw_events = self.sim.run()
        self.injector = AttackInjector(seed=42)

    def test_dos_attack(self) -> None:
        spec = AttackSpec(
            attack_type="DoS",
            target_sensor="ultrasonic",
            start_time=10.0,
            duration=3.0,
            intensity=1.5,
        )
        injected = self.injector.inject(self.raw_events, [spec])

        # Find DoS packets
        dos_events = [e for e in injected if e.is_attack == 1 and e.attack_type == "DoS"]
        self.assertGreater(len(dos_events), 15, "DoS should inject a significant burst of packets")

        for e in dos_events:
            self.assertEqual(e.sensor_id, "ultrasonic")
            self.assertEqual(e.attack_target, "ultrasonic")
            self.assertEqual(e.event_type, "FLOOD")
            self.assertEqual(e.output_symbol, "o6")
            self.assertGreaterEqual(e.timestamp, 10.0)
            self.assertLessEqual(e.timestamp, 13.5)

    def test_replay_attack(self) -> None:
        spec = AttackSpec(
            attack_type="Replay",
            target_sensor="motion",
            start_time=12.0,
            duration=6.0,
            params={"delay_offset": 4.0},
        )
        injected = self.injector.inject(self.raw_events, [spec])

        replay_events = [e for e in injected if e.is_attack == 1 and e.attack_type == "Replay"]
        self.assertGreater(len(replay_events), 0, "Replay attack should re-inject candidate packets")

        for e in replay_events:
            self.assertEqual(e.sensor_id, "motion")
            self.assertIsNotNone(e.original_timestamp)
            self.assertGreater(e.timestamp, e.original_timestamp)
            self.assertEqual(e.output_symbol, "o6")  # Stale/duplicate packet rejected

    def test_mitm_attack(self) -> None:
        spec = AttackSpec(
            attack_type="MITM",
            target_sensor="temperature",
            start_time=8.0,
            duration=5.0,
        )
        injected = self.injector.inject(self.raw_events, [spec])

        mitm_events = [e for e in injected if e.is_attack == 1 and e.attack_type == "MITM"]
        self.assertGreater(len(mitm_events), 0, "MITM attack should tamper with active packets")

        for e in mitm_events:
            self.assertEqual(e.sensor_id, "temperature")
            self.assertIn("tampered_by", e.payload)
            self.assertEqual(e.payload["tampered_by"], "mitm_proxy")
            self.assertIsNotNone(e.original_payload)
            self.assertNotEqual(e.payload, e.original_payload)

    def test_data_falsification_attack(self) -> None:
        spec = AttackSpec(
            attack_type="DataFalsification",
            target_sensor="temperature",
            start_time=15.0,
            duration=5.0,
        )
        injected = self.injector.inject(self.raw_events, [spec])

        dfa_events = [e for e in injected if e.is_attack == 1 and e.attack_type == "DataFalsification"]
        self.assertGreater(len(dfa_events), 0, "DFA should corrupt telemetry packets")

        for e in dfa_events:
            self.assertEqual(e.sensor_id, "temperature")
            self.assertEqual(e.payload["temperature"], 185.5)  # Severe physical impossibility
            self.assertEqual(e.output_symbol, "o6")
            self.assertIsNotNone(e.original_payload)

    def test_message_delay_attack(self) -> None:
        spec = AttackSpec(
            attack_type="Delay",
            target_sensor="ultrasonic",
            start_time=6.0,
            duration=4.0,
            params={"delay_sec": 3.5},
        )
        injected = self.injector.inject(self.raw_events, [spec])

        delay_events = [e for e in injected if e.is_attack == 1 and e.attack_type == "Delay"]
        self.assertGreater(len(delay_events), 0, "Delay attack should affect packets in window")

        for e in delay_events:
            self.assertEqual(e.sensor_id, "ultrasonic")
            self.assertIsNotNone(e.original_timestamp)
            delta = e.timestamp - e.original_timestamp
            self.assertAlmostEqual(delta, 3.5, places=2)

    def test_rfid_spoofing_attack(self) -> None:
        spec = AttackSpec(
            attack_type="IdentitySpoofing",
            target_sensor="rfid",
            start_time=5.0,
            duration=15.0,
        )
        injected = self.injector.inject(self.raw_events, [spec])

        spoof_events = [e for e in injected if e.is_attack == 1 and e.attack_type == "IdentitySpoofing"]
        self.assertGreater(len(spoof_events), 0, "RFID spoofing should alter unauthorized card scans")

        for e in spoof_events:
            self.assertEqual(e.sensor_id, "rfid")
            self.assertEqual(e.payload["card_uid"], "SPOOFED_UNAUTH_UID")
            self.assertEqual(e.payload["status"], "authorized")
            self.assertEqual(e.output_symbol, "o11")  # Granted access output

    def test_multi_attack_interleaved_scenario(self) -> None:
        specs = [
            AttackSpec("DoS", "ultrasonic", start_time=5.0, duration=2.0),
            AttackSpec("MITM", "temperature", start_time=10.0, duration=3.0),
            AttackSpec("DataFalsification", "motion", start_time=15.0, duration=3.0),
            AttackSpec("Delay", "ultrasonic", start_time=20.0, duration=2.0, params={"delay_sec": 4.0}),
            AttackSpec("IdentitySpoofing", "rfid", start_time=8.0, duration=10.0),
        ]
        injected = self.injector.inject(self.raw_events, specs)

        # Monotonicity check across all events
        for i in range(len(injected) - 1):
            self.assertLessEqual(injected[i].timestamp, injected[i + 1].timestamp)

        # Verify attacks are present and normal packets remain uncorrupted
        attack_types = {e.attack_type for e in injected}
        self.assertIn("Normal", attack_types)
        self.assertIn("DoS", attack_types)
        self.assertIn("MITM", attack_types)
        self.assertIn("DataFalsification", attack_types)
        self.assertIn("Delay", attack_types)
        self.assertIn("IdentitySpoofing", attack_types)

        # Normal events must have is_attack=0
        for e in injected:
            if e.attack_type == "Normal":
                self.assertEqual(e.is_attack, 0)
            else:
                self.assertEqual(e.is_attack, 1)

    def test_anti_leakage_feature_separation_across_attacks(self) -> None:
        specs = [
            AttackSpec("DoS", "ultrasonic", start_time=5.0, duration=2.0),
            AttackSpec("MITM", "temperature", start_time=10.0, duration=3.0),
            AttackSpec("IdentitySpoofing", "rfid", start_time=15.0, duration=5.0),
        ]
        injected = self.injector.inject(self.raw_events, specs)

        quarantined_keys = ["is_attack", "attack_type", "attack_target", "fault_location", "original_payload", "original_timestamp"]
        for e in injected:
            obs = e.to_observable_dict()
            for qk in quarantined_keys:
                self.assertNotIn(qk, obs, f"Leakage detected: {qk} exposed in observable dictionary")


if __name__ == "__main__":
    unittest.main()

