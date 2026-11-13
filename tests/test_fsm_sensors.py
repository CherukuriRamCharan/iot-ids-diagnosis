"""Unit tests for Stage 1 Four-Sensor Formal TFSM Models and Mutation Testing.

Verifies:
1. Motion Sensor TFSM (models_fsm/motion_sensor.py) - 14 mutants
2. Temperature Sensor TFSM (models_fsm/temperature_sensor.py) - 15 mutants
3. Ultrasonic Sensor TFSM (models_fsm/ultrasonic_sensor.py) - 15 mutants
4. RFID Reader TFSM (models_fsm/rfid_sensor.py) - 15 mutants
5. Total: 59 mutants verified across all four sensors with 100% kill rate.
"""
from __future__ import annotations

import unittest

from models_fsm import motion_sensor, rfid_sensor, temperature_sensor, ultrasonic_sensor
from models_fsm.tfsm import signature
from models_fsm.verify import kill_matrix_for


class TestMotionSensorFSM(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = motion_sensor
        self.orig = self.mod.build_original()

    def test_initial_state_and_device(self) -> None:
        self.assertEqual(self.orig.init, "s1")
        self.assertEqual(self.mod.DEVICE, "motion_sensor")
        self.assertGreater(len(self.orig.transitions), 15)

    def test_startup_and_motion_trace(self) -> None:
        steps = self.mod.parse("Delay(2)-i2-delay(2)-i4")
        events = self.orig.run(steps)
        outputs = [e.sym for e in events if e.direction == "out"]
        self.assertIn("o2", outputs)
        self.assertIn("o4", outputs)

    def test_invalid_input_yields_o6(self) -> None:
        steps = self.mod.parse("Delay(2)-i2-delay(2)-i4-i4")
        events = self.orig.run(steps)
        last_out = [e for e in events if e.direction == "out"][-1]
        self.assertEqual(last_out.sym, "o6")

    def test_mutant_count(self) -> None:
        self.assertEqual(len(self.mod.MUTANTS), 14)

    def test_all_mutants_killed(self) -> None:
        rows = kill_matrix_for(self.mod)
        self.assertEqual(len(rows), 14)
        for m, killed_by in rows:
            self.assertIn(
                m.test,
                killed_by,
                f"Motion mutant {m.mid} was NOT killed by designated test {m.test}",
            )


class TestTemperatureSensorFSM(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = temperature_sensor
        self.orig = self.mod.build_original()

    def test_initial_state_and_device(self) -> None:
        self.assertEqual(self.orig.init, "s1")
        self.assertEqual(self.mod.DEVICE, "temperature_sensor")

    def test_temp_normal_and_alert_outputs(self) -> None:
        # Normal temp: i11 -> o11
        steps_norm = self.mod.parse("Delay(2)-i2-delay(2)-i4-i10-delay(2)-delay(2)-i11")
        events_norm = self.orig.run(steps_norm)
        outs_norm = [e.sym for e in events_norm if e.direction == "out"]
        self.assertIn("o11", outs_norm)

        # High temp alert: i12 -> o12
        steps_alert = self.mod.parse("Delay(2)-i2-delay(2)-i4-i10-delay(2)-delay(2)-i12")
        events_alert = self.orig.run(steps_alert)
        outs_alert = [e.sym for e in events_alert if e.direction == "out"]
        self.assertIn("o12", outs_alert)

    def test_mutant_count(self) -> None:
        self.assertEqual(len(self.mod.MUTANTS), 15)

    def test_bd_s9_and_sda_s9_killed(self) -> None:
        # Explicit test for BD_s9 killed by BD5
        bd_s9 = self.mod.MUTANT_BY_ID["BD_s9"]
        mm_bd = bd_s9.build(self.orig)
        steps_bd5 = self.mod.steps_of("BD5")
        sig_orig_bd5 = signature(self.orig.run(steps_bd5, self.mod.TRAILING))
        sig_mut_bd5 = signature(mm_bd.run(steps_bd5, self.mod.TRAILING))
        self.assertNotEqual(sig_orig_bd5, sig_mut_bd5, "BD_s9 must differ on BD5")

        # Explicit test for BD_SDA_s9 killed by SDA
        sda = self.mod.MUTANT_BY_ID["BD_SDA_s9"]
        mm_sda = sda.build(self.orig)
        steps_sda = self.mod.steps_of("SDA")
        sig_orig_sda = signature(self.orig.run(steps_sda, self.mod.TRAILING))
        sig_mut_sda = signature(mm_sda.run(steps_sda, self.mod.TRAILING))
        self.assertNotEqual(sig_orig_sda, sig_mut_sda, "BD_SDA_s9 must differ on SDA")

    def test_all_mutants_killed(self) -> None:
        rows = kill_matrix_for(self.mod)
        self.assertEqual(len(rows), 15)
        for m, killed_by in rows:
            self.assertIn(
                m.test,
                killed_by,
                f"Temperature mutant {m.mid} was NOT killed by designated test {m.test}",
            )


class TestUltrasonicSensorFSM(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = ultrasonic_sensor
        self.orig = self.mod.build_original()

    def test_initial_state_and_device(self) -> None:
        self.assertEqual(self.orig.init, "s1")
        self.assertEqual(self.mod.DEVICE, "ultrasonic_sensor")

    def test_distance_and_proximity_outputs(self) -> None:
        # Normal distance: i11 -> o11
        steps_dist = self.mod.parse("Delay(2)-i2-delay(2)-i4-i10-delay(2)-delay(2)-i11")
        events_dist = self.orig.run(steps_dist)
        outs_dist = [e.sym for e in events_dist if e.direction == "out"]
        self.assertIn("o11", outs_dist)

        # Obstacle alert: i12 -> o12
        steps_obs = self.mod.parse("Delay(2)-i2-delay(2)-i4-i10-delay(2)-delay(2)-i12")
        events_obs = self.orig.run(steps_obs)
        outs_obs = [e.sym for e in events_obs if e.direction == "out"]
        self.assertIn("o12", outs_obs)

    def test_mutant_count(self) -> None:
        self.assertEqual(len(self.mod.MUTANTS), 15)

    def test_bd_s9_and_sda_s9_killed(self) -> None:
        # Explicit test for BD_s9 killed by BD5
        bd_s9 = self.mod.MUTANT_BY_ID["BD_s9"]
        mm_bd = bd_s9.build(self.orig)
        steps_bd5 = self.mod.steps_of("BD5")
        sig_orig_bd5 = signature(self.orig.run(steps_bd5, self.mod.TRAILING))
        sig_mut_bd5 = signature(mm_bd.run(steps_bd5, self.mod.TRAILING))
        self.assertNotEqual(sig_orig_bd5, sig_mut_bd5, "BD_s9 must differ on BD5")

        # Explicit test for BD_SDA_s9 killed by SDA
        sda = self.mod.MUTANT_BY_ID["BD_SDA_s9"]
        mm_sda = sda.build(self.orig)
        steps_sda = self.mod.steps_of("SDA")
        sig_orig_sda = signature(self.orig.run(steps_sda, self.mod.TRAILING))
        sig_mut_sda = signature(mm_sda.run(steps_sda, self.mod.TRAILING))
        self.assertNotEqual(sig_orig_sda, sig_mut_sda, "BD_SDA_s9 must differ on SDA")

    def test_all_mutants_killed(self) -> None:
        rows = kill_matrix_for(self.mod)
        self.assertEqual(len(rows), 15)
        for m, killed_by in rows:
            self.assertIn(
                m.test,
                killed_by,
                f"Ultrasonic mutant {m.mid} was NOT killed by designated test {m.test}",
            )


class TestRFIDSensorFSM(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = rfid_sensor
        self.orig = self.mod.build_original()

    def test_initial_state_and_device(self) -> None:
        self.assertEqual(self.orig.init, "s1")
        self.assertEqual(self.mod.DEVICE, "rfid_sensor")

    def test_authorized_and_unauthorized_scans(self) -> None:
        steps_grant = self.mod.parse("Delay(2)-i2-delay(2)-i4-i10-delay(2)-delay(2)-i11")
        events_grant = self.orig.run(steps_grant)
        outs_grant = [e.sym for e in events_grant if e.direction == "out"]
        self.assertIn("o11", outs_grant)

        steps_deny = self.mod.parse("Delay(2)-i2-delay(2)-i4-i10-delay(2)-delay(2)-i12")
        events_deny = self.orig.run(steps_deny)
        outs_deny = [e.sym for e in events_deny if e.direction == "out"]
        self.assertIn("o12", outs_deny)

    def test_spoof_mutant_changes_denied_to_granted(self) -> None:
        spoof_mut = self.mod.MUTANT_BY_ID["SPOOF_t23"]
        mut_machine = spoof_mut.build(self.orig)
        steps_c = self.mod.steps_of("C")
        orig_sig = signature(self.orig.run(steps_c, self.mod.TRAILING))
        mut_sig = signature(mut_machine.run(steps_c, self.mod.TRAILING))
        self.assertNotEqual(orig_sig, mut_sig)
        self.assertIn("out:o11", mut_sig)

    def test_mutant_count(self) -> None:
        self.assertEqual(len(self.mod.MUTANTS), 15)

    def test_all_mutants_killed(self) -> None:
        rows = kill_matrix_for(self.mod)
        self.assertEqual(len(rows), 15)
        for m, killed_by in rows:
            self.assertIn(
                m.test,
                killed_by,
                f"RFID mutant {m.mid} was NOT killed by designated test {m.test}",
            )


class TestTotalMutantKillSummary(unittest.TestCase):
    def test_total_59_mutants_all_killed(self) -> None:
        total_mutants = 0
        total_killed = 0
        modules = [motion_sensor, temperature_sensor, ultrasonic_sensor, rfid_sensor]
        expected_counts = [14, 15, 15, 15]

        for mod, exp_count in zip(modules, expected_counts):
            rows = kill_matrix_for(mod)
            self.assertEqual(len(rows), exp_count)
            total_mutants += len(rows)
            for m, killed_by in rows:
                if m.test in killed_by:
                    total_killed += 1

        self.assertEqual(total_mutants, 59, "Total mutants across 4 sensors must be 59")
        self.assertEqual(total_killed, 59, "All 59 mutants must be killed (100% kill rate)")


if __name__ == "__main__":
    unittest.main()
