"""Unit tests for Stage 4 Formal FSM-Based Diagnosis & Fault Localization Engine."""

from __future__ import annotations

import copy
import unittest

from diagnosis_loc.unified_diagnosis import UnifiedDiagnosisEngine
from simulation.attack_injector import AttackInjector, AttackSpec
from simulation.unified_simulator import UnifiedSimulator


class TestUnifiedDiagnosis(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = UnifiedDiagnosisEngine()
        self.injector = AttackInjector(seed=42)

    def test_normal_sequences_all_four_sensors(self) -> None:
        sim = UnifiedSimulator(duration=20.0, seed=42)
        events = sim.run()

        report = self.engine.diagnose_events(events)

        self.assertFalse(report.overall_anomaly_detected, "Normal sequence should not trigger overall anomaly")
        self.assertEqual(report.affected_sensors, [])
        self.assertEqual(report.primary_predicted_attack, "Normal")

        # All 4 sensors must have 100% formal FSM conformance
        for sid in ("motion", "temperature", "ultrasonic", "rfid"):
            s_rep = report.sensor_reports[sid]
            self.assertFalse(s_rep.anomaly_detected, f"Sensor {sid} falsely flagged as anomaly")
            self.assertTrue(s_rep.fsm_conformance, f"Sensor {sid} failed FSM conformance on normal run")
            self.assertEqual(s_rep.anomalous_events_count, 0)
            self.assertEqual(s_rep.diagnosis_confidence, 1.0)

    def test_detection_of_invalid_fsm_transitions(self) -> None:
        sim = UnifiedSimulator(duration=10.0, seed=42)
        events = sim.run()
        obs_events = [e.to_observable_dict() for e in events]

        # Inject an invalid output symbol directly into a temperature event
        for ev in obs_events:
            if ev["sensor_id"] == "temperature" and ev["event_type"] == "PUBLISH":
                ev["output_symbol"] = "o6"  # Unexpected error output symbol on normal read
                break

        report = self.engine.diagnose_events(obs_events)
        temp_rep = report.sensor_reports["temperature"]

        self.assertTrue(temp_rep.anomaly_detected)
        self.assertFalse(temp_rep.fsm_conformance, "Invalid output must violate FSM conformance")
        self.assertIn("temperature", report.affected_sensors)
        self.assertGreater(len(temp_rep.evidence), 0)

    def test_sensor_specific_diagnosis(self) -> None:
        sim = UnifiedSimulator(duration=20.0, seed=42)
        raw_events = sim.run()

        # Attack ONLY ultrasonic sensor with DoS
        spec = AttackSpec("DoS", "ultrasonic", start_time=8.0, duration=2.5, intensity=1.5)
        attack_events = self.injector.inject(raw_events, [spec])

        report = self.engine.diagnose_events(attack_events)

        self.assertTrue(report.overall_anomaly_detected)
        self.assertEqual(report.affected_sensors, ["ultrasonic"], "Only ultrasonic should be affected")
        self.assertEqual(report.sensor_reports["ultrasonic"].predicted_attack_category, "DoS")

        # Other three sensors must remain completely benign
        for sid in ("motion", "temperature", "rfid"):
            self.assertFalse(report.sensor_reports[sid].anomaly_detected, f"Sensor {sid} falsely affected")

    def test_dos_evidence_and_diagnosis(self) -> None:
        sim = UnifiedSimulator(duration=20.0, seed=42)
        raw_events = sim.run()

        spec = AttackSpec("DoS", "motion", start_time=6.0, duration=3.0, intensity=1.5)
        events = self.injector.inject(raw_events, [spec])

        report = self.engine.diagnose_events(events)
        motion_rep = report.sensor_reports["motion"]

        self.assertTrue(motion_rep.anomaly_detected)
        self.assertEqual(motion_rep.predicted_attack_category, "DoS")
        self.assertIn("dos_queue_exhaustion", motion_rep.candidate_faults)
        self.assertGreaterEqual(motion_rep.diagnosis_confidence, 0.90)

    def test_replay_evidence_and_diagnosis(self) -> None:
        sim = UnifiedSimulator(duration=25.0, seed=42)
        raw_events = sim.run()

        spec = AttackSpec("Replay", "temperature", start_time=10.0, duration=6.0, params={"delay_offset": 4.0})
        events = self.injector.inject(raw_events, [spec])

        report = self.engine.diagnose_events(events)
        temp_rep = report.sensor_reports["temperature"]

        self.assertTrue(temp_rep.anomaly_detected)
        self.assertEqual(temp_rep.predicted_attack_category, "Replay")
        self.assertIn("replayed_stale_sequence", temp_rep.candidate_faults)

    def test_mitm_evidence_and_diagnosis(self) -> None:
        sim = UnifiedSimulator(duration=20.0, seed=42)
        raw_events = sim.run()

        spec = AttackSpec("MITM", "ultrasonic", start_time=7.0, duration=5.0)
        events = self.injector.inject(raw_events, [spec])

        report = self.engine.diagnose_events(events)
        ultra_rep = report.sensor_reports["ultrasonic"]

        self.assertTrue(ultra_rep.anomaly_detected)
        self.assertEqual(ultra_rep.predicted_attack_category, "MITM")
        self.assertIn("mitm_intercept_tamper", ultra_rep.candidate_faults)

    def test_data_falsification_evidence_and_diagnosis(self) -> None:
        sim = UnifiedSimulator(duration=20.0, seed=42)
        raw_events = sim.run()

        spec = AttackSpec("DataFalsification", "temperature", start_time=8.0, duration=5.0)
        events = self.injector.inject(raw_events, [spec])

        report = self.engine.diagnose_events(events)
        temp_rep = report.sensor_reports["temperature"]

        self.assertTrue(temp_rep.anomaly_detected)
        self.assertEqual(temp_rep.predicted_attack_category, "DataFalsification")
        self.assertIn("dfa_sensor_falsification", temp_rep.candidate_faults)

    def test_delay_evidence_and_diagnosis(self) -> None:
        sim = UnifiedSimulator(duration=20.0, seed=42)
        raw_events = sim.run()

        spec = AttackSpec("Delay", "ultrasonic", start_time=6.0, duration=4.0, params={"delay_sec": 3.5})
        events = self.injector.inject(raw_events, [spec])

        report = self.engine.diagnose_events(events)
        ultra_rep = report.sensor_reports["ultrasonic"]

        self.assertTrue(ultra_rep.anomaly_detected)
        self.assertEqual(ultra_rep.predicted_attack_category, "Delay")
        self.assertIn("delay_timeout_exceeded", ultra_rep.candidate_faults)

    def test_rfid_spoofing_evidence_and_diagnosis(self) -> None:
        sim = UnifiedSimulator(duration=25.0, seed=42)
        raw_events = sim.run()

        spec = AttackSpec("IdentitySpoofing", "rfid", start_time=5.0, duration=15.0)
        events = self.injector.inject(raw_events, [spec])

        report = self.engine.diagnose_events(events)
        rfid_rep = report.sensor_reports["rfid"]

        self.assertTrue(rfid_rep.anomaly_detected)
        self.assertFalse(rfid_rep.fsm_conformance, "Spoofing must violate formal conformance")
        self.assertEqual(rfid_rep.predicted_attack_category, "IdentitySpoofing")
        self.assertIn("rfid_uid_spoofing_t23", rfid_rep.candidate_faults)
        self.assertGreaterEqual(rfid_rep.diagnosis_confidence, 0.90)

    def test_multiple_simultaneous_attack_scenarios(self) -> None:
        sim = UnifiedSimulator(duration=25.0, seed=42)
        raw_events = sim.run()

        specs = [
            AttackSpec("DoS", "motion", start_time=6.0, duration=3.0),
            AttackSpec("IdentitySpoofing", "rfid", start_time=8.0, duration=10.0),
        ]
        events = self.injector.inject(raw_events, specs)

        report = self.engine.diagnose_events(events)

        self.assertTrue(report.overall_anomaly_detected)
        self.assertIn("motion", report.affected_sensors)
        self.assertIn("rfid", report.affected_sensors)
        self.assertEqual(report.sensor_reports["motion"].predicted_attack_category, "DoS")
        self.assertEqual(report.sensor_reports["rfid"].predicted_attack_category, "IdentitySpoofing")

    def test_no_ground_truth_leakage_in_diagnosis(self) -> None:
        sim = UnifiedSimulator(duration=20.0, seed=42)
        raw_events = sim.run()
        specs = [AttackSpec("DoS", "ultrasonic", start_time=6.0, duration=3.0)]
        events = self.injector.inject(raw_events, specs)

        # Create dictionaries with ground-truth completely deleted
        clean_dicts = []
        for e in events:
            d = e.to_observable_dict()
            self.assertNotIn("is_attack", d)
            self.assertNotIn("attack_type", d)
            self.assertNotIn("attack_target", d)
            self.assertNotIn("fault_location", d)
            clean_dicts.append(d)

        report_from_objects = self.engine.diagnose_events(events)
        report_from_clean = self.engine.diagnose_events(clean_dicts)

        self.assertEqual(report_from_objects.affected_sensors, report_from_clean.affected_sensors)
        self.assertEqual(report_from_objects.primary_predicted_attack, report_from_clean.primary_predicted_attack)

    def test_deterministic_results(self) -> None:
        sim = UnifiedSimulator(duration=20.0, seed=42)
        raw_events = sim.run()
        specs = [AttackSpec("DataFalsification", "temperature", start_time=6.0, duration=4.0)]
        events = self.injector.inject(raw_events, specs)

        rep1 = self.engine.diagnose_events(events)
        rep2 = self.engine.diagnose_events(events)

        self.assertEqual(rep1.affected_sensors, rep2.affected_sensors)
        self.assertEqual(rep1.primary_predicted_attack, rep2.primary_predicted_attack)
        self.assertEqual(len(rep1.event_diagnoses), len(rep2.event_diagnoses))

    def test_ambiguous_diagnosis_handling(self) -> None:
        # Construct an artificial ambiguous event: delayed frame with unknown error
        ambiguous_event = {
            "timestamp": 10.5,
            "sensor_id": "ultrasonic",
            "topic": "distance/ultrasonic1",
            "event_type": "PUBLISH",
            "payload": {"distance_cm": 150.0},
            "sequence_number": 12,
            "qos": 0,
            "payload_size": 48,
            "observed_state": "s9",
            "output_symbol": "o6",  # Discrepancy
            "inter_arrival": 3.8,   # Delay violation as well
        }
        report = self.engine.diagnose_events([ambiguous_event])
        u_rep = report.sensor_reports["ultrasonic"]

        self.assertTrue(u_rep.anomaly_detected)
        # Should record multiple candidate faults instead of pretending single-fault certainty
        self.assertGreaterEqual(len(u_rep.candidate_faults), 2)

    def test_stale_sequence_with_normal_payload_is_replay(self) -> None:
        """A stale sequence number with otherwise normal payload must be diagnosed as Replay."""
        events = [
            {
                "timestamp": 2.0,
                "sensor_id": "temperature",
                "topic": "home/sensor/temperature1",
                "event_type": "PUBLISH",
                "payload": {"temperature": 21.5, "alert": False},
                "sequence_number": 1,
                "qos": 0,
                "payload_size": 48,
                "observed_state": "s9",
                "output_symbol": "o11",
            },
            {
                "timestamp": 4.0,
                "sensor_id": "temperature",
                "topic": "home/sensor/temperature1",
                "event_type": "PUBLISH",
                "payload": {"temperature": 22.0, "alert": False},
                "sequence_number": 2,
                "qos": 0,
                "payload_size": 48,
                "observed_state": "s9",
                "output_symbol": "o11",
            },
            {
                "timestamp": 5.0,
                "sensor_id": "temperature",
                "topic": "home/sensor/temperature1",
                "event_type": "PUBLISH",
                "payload": {"temperature": 21.5, "alert": False},
                "sequence_number": 1,  # Stale / duplicate sequence number
                "qos": 0,
                "payload_size": 48,
                "observed_state": "s9",
                "output_symbol": "o6",  # Receiver rejects replay
            },
        ]
        report = self.engine.diagnose_events(events)
        temp_rep = report.sensor_reports["temperature"]

        self.assertTrue(temp_rep.anomaly_detected)
        self.assertEqual(temp_rep.predicted_attack_category, "Replay")
        self.assertIn("replayed_stale_sequence", temp_rep.candidate_faults)
        self.assertNotIn("dfa_sensor_falsification", temp_rep.candidate_faults)

    def test_physically_implausible_payload_without_replay_is_data_falsification(self) -> None:
        """Genuinely modified physically implausible payload on fresh sequence must be diagnosed as DataFalsification."""
        events = [
            {
                "timestamp": 2.0,
                "sensor_id": "temperature",
                "topic": "home/sensor/temperature1",
                "event_type": "PUBLISH",
                "payload": {"temperature": 21.5, "alert": False},
                "sequence_number": 1,
                "qos": 0,
                "payload_size": 48,
                "observed_state": "s9",
                "output_symbol": "o11",
            },
            {
                "timestamp": 4.0,
                "sensor_id": "temperature",
                "topic": "home/sensor/temperature1",
                "event_type": "PUBLISH",
                "payload": {"temperature": 185.0, "alert": True},  # Physically impossible > 70°C
                "sequence_number": 2,  # Strictly monotonic fresh sequence
                "qos": 0,
                "payload_size": 48,
                "observed_state": "s9",
                "output_symbol": "o6",
            },
        ]
        report = self.engine.diagnose_events(events)
        temp_rep = report.sensor_reports["temperature"]

        self.assertTrue(temp_rep.anomaly_detected)
        self.assertEqual(temp_rep.predicted_attack_category, "DataFalsification")
        self.assertIn("dfa_sensor_falsification", temp_rep.candidate_faults)
        self.assertNotIn("replayed_stale_sequence", temp_rep.candidate_faults)

    def test_repeated_normal_payload_with_fresh_sequence_is_benign(self) -> None:
        """Repeated identical payload with monotonic sequence and normal timing must NOT be flagged as attack."""
        events = [
            {
                "timestamp": 2.0,
                "sensor_id": "temperature",
                "topic": "home/sensor/temperature1",
                "event_type": "PUBLISH",
                "payload": {"temperature": 22.0, "alert": False},
                "sequence_number": 1,
                "qos": 0,
                "payload_size": 48,
                "observed_state": "s9",
                "output_symbol": "o11",
            },
            {
                "timestamp": 4.0,
                "sensor_id": "temperature",
                "topic": "home/sensor/temperature1",
                "event_type": "PUBLISH",
                "payload": {"temperature": 22.0, "alert": False},  # Identical reading
                "sequence_number": 2,  # Fresh sequential number
                "qos": 0,
                "payload_size": 48,
                "observed_state": "s9",
                "output_symbol": "o11",
            },
            {
                "timestamp": 6.0,
                "sensor_id": "temperature",
                "topic": "home/sensor/temperature1",
                "event_type": "PUBLISH",
                "payload": {"temperature": 22.0, "alert": False},  # Identical reading
                "sequence_number": 3,  # Fresh sequential number
                "qos": 0,
                "payload_size": 48,
                "observed_state": "s9",
                "output_symbol": "o11",
            },
        ]
        report = self.engine.diagnose_events(events)
        temp_rep = report.sensor_reports["temperature"]

        self.assertFalse(temp_rep.anomaly_detected, "Repeated normal payload must remain benign")
        self.assertTrue(temp_rep.fsm_conformance)
        self.assertEqual(temp_rep.predicted_attack_category, "Normal")
        self.assertEqual(temp_rep.anomalous_events_count, 0)


if __name__ == "__main__":
    unittest.main()

