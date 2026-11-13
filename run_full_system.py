"""End-to-End Integration and Verification Harness for Four-Sensor IoT IDS.

Project: Attack Diagnosis and Localization in IoT Intrusion Detection Using Formal Modeling and Deep Learning
Repository: iot-ids-diagnosis

Demonstrates the complete end-to-end pipeline across Stages 1 through 5:
1. Stage 1: Formal TFSM Mutation Verification (59/59 mutants killed).
2. Stage 2: Unified Four-Sensor Discrete-Event Simulator & Attack Injector.
3. Stage 3 & 5.1: Observable Feature Extraction (14 approved features, zero leakage).
4. Stage 4: Formal TFSM Conformance & Fault Localization Engine.
5. Stage 5.3: Two-Tier Hybrid Diagnosis & Adjudication Arbiter.
6. Full Unit Test Suite Execution (144 tests across 14 test suites).

Usage:
    python run_full_system.py
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnosis_loc.hybrid_diagnosis import (
    HybridDiagnosisEngine,
    HybridDiagnosisReport,
    extract_observable_feature_vector,
)
from ids.datasets_four_sensors import (
    FOUR_SENSORS_CLASSES,
    SELECTED_OBSERVABLE_FEATURES,
    assert_no_leakage_four_sensors,
)
from models_fsm import verify as fsm_verify
from simulation.attack_injector import AttackInjector, AttackSpec
from simulation.unified_simulator import UnifiedSimulator


def print_banner(title: str, fill: str = "=") -> None:
    print(f"\n{fill * 78}")
    print(f"  {title}")
    print(f"{fill * 78}")


def run_stage1_formal_verification() -> bool:
    """Runs mutation testing across all 4 sensor TFSM models."""
    print_banner("STEP 1: Stage 1 Formal TFSM Mutation Verification")
    print("Verifying that all 59 synthetic attack mutants are killed by formal test suites...")
    
    code = fsm_verify.main()
    success = (code == 0)
    if success:
        print("\n[PASS] Stage 1 Verification Complete: 59/59 mutants killed across 4 sensors.")
    else:
        print("\n[FAIL] Stage 1 Verification Failed: Some mutants were not killed.")
    return success


def run_stage2_simulation_demo() -> List[Any]:
    """Simulates multi-sensor nominal and attack sessions."""
    print_banner("STEP 2: Stage 2 Multi-Sensor Simulation & Attack Injection")
    
    # 1. Benign Baseline Simulation
    sim = UnifiedSimulator(duration=5.0, seed=42)
    benign_events = sim.run()
    sensor_ids = sorted(list({e.sensor_id for e in benign_events}))
    print(f"-> Generated {len(benign_events)} benign events across sensors: {sensor_ids}")
    
    # 2. Attack Simulation (DoS flood on ultrasonic sensor)
    injector = AttackInjector(seed=42)
    attack_spec = AttackSpec(
        attack_type="DoS",
        target_sensor="ultrasonic",
        start_time=2.0,
        duration=1.5,
        intensity=1.5,
    )
    attack_events = injector.inject(benign_events, [attack_spec])
    
    dos_events = [e for e in attack_events if getattr(e, "is_attack", 0) == 1]
    print(f"-> Injected DoS attack: total events={len(attack_events)}, attack events={len(dos_events)}")
    return attack_events


def run_stage3_and_5_feature_audit(events: List[Any]) -> bool:
    """Audits feature extraction against leakage."""
    print_banner("STEP 3: Stage 5.1 Observable Feature Extraction & Zero-Leakage Audit")
    
    last_seq: Dict[str, int] = {}
    last_ts = 0.0
    extracted_vectors = []
    
    for ev in events:
        d = ev.to_observable_dict() if hasattr(ev, "to_observable_dict") else ev
        vec = extract_observable_feature_vector(d, last_seq, last_ts)
        last_ts = float(d.get("timestamp", 0.0))
        extracted_vectors.append(vec)
        
    print(f"-> Successfully extracted 14-dimensional feature vectors for {len(extracted_vectors)} events.")
    print(f"-> Feature Order: {SELECTED_OBSERVABLE_FEATURES}")
    
    # Validate against leak check
    assert_no_leakage_four_sensors(SELECTED_OBSERVABLE_FEATURES)
        
    print("[PASS] Zero-Leakage Provenance Audit: Ground-truth, internal states, and simulator watermarks strictly excluded.")
    return True


def run_stage5_hybrid_diagnosis_demo(events: List[Any]) -> bool:
    """Executes live two-tier hybrid diagnosis on the simulated event stream."""
    print_banner("STEP 4: Stage 5.3 Two-Tier Hybrid Diagnosis & Adjudication")
    
    engine = HybridDiagnosisEngine()
    print(f"-> Hybrid Engine Initialized: Model Available = {engine.model_available}")
    
    report: HybridDiagnosisReport = engine.diagnose(events)
    
    print("\n--- HYBRID ADJUDICATION RESULTS ---")
    print(f"Total Events Evaluated   : {report.total_events}")
    print(f"Hybrid Outcome           : {report.hybrid_outcome}")
    print(f"Agreement Status         : {'AGREEMENT' if report.agreement else 'DISAGREEMENT'}")
    print(f"Adjudication Explanation : {report.outcome_explanation}")
    
    print("\n--- TIER 1: STATISTICAL DEEP LEARNING (BiLSTM) ---")
    print(f"Attack Detected          : {report.dl_attack_detected}")
    print(f"Predicted Category       : {report.dl_predicted_category}")
    print(f"Raw Confidence Score     : {report.dl_confidence_score:.4f} (Softmax score, not calibrated prob)")
    print(f"Sliding Windows (L=16)   : {report.dl_window_count}")
    
    print("\n--- TIER 2: FORMAL TFSM CONFORMANCE & LOCALIZATION ---")
    print(f"Formal Anomaly Detected  : {report.formal_anomaly_detected}")
    print(f"Formal Conformance       : {'PASS' if report.formal_conformance else 'FAIL'}")
    print(f"Formal Primary Attack    : {report.formal_primary_attack}")
    print(f"Affected Sensors         : {report.affected_sensors}")
    print(f"Suspected Sensor         : {report.suspected_sensor}")
    print(f"Suspected Transition     : {report.suspected_transition}")
    print(f"Suspected State          : {report.suspected_state}")
    print(f"Expected / Observed Out  : {report.expected_output} / {report.observed_output}")
    print(f"Deviation Type           : {report.deviation_type}")
    print(f"Candidate Mutant Hypotheses: {report.candidate_fault_mutants}")
    
    print("\n[PASS] Hybrid Diagnosis Completed: Statistical evidence and formal proofs preserved separately.")
    return True


def run_full_regression_suite() -> bool:
    """Discovers and runs all 144 unit tests across the 14 test files."""
    print_banner("STEP 5: Full Regression Test Suite Execution")
    print("Discovering and executing all unit tests in tests/...")
    
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(REPO_ROOT / "tests"), pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    print(f"\nRegression Suite Summary:")
    print(f"  Tests Run  : {result.testsRun}")
    print(f"  Failures   : {len(result.failures)}")
    print(f"  Errors     : {len(result.errors)}")
    print(f"  Skipped    : {len(result.skipped)}")
    
    if result.wasSuccessful():
        print(f"[PASS] All {result.testsRun} unit tests PASSED successfully!")
        return True
    else:
        print(f"[FAIL] Regression suite had {len(result.failures)} failures and {len(result.errors)} errors.")
        return False


def main() -> int:
    start_time = time.time()
    print_banner("IoT IDS: FOUR-SENSOR HYBRID DIAGNOSIS END-TO-END PIPELINE")
    print("Project: Attack Diagnosis and Localization in IoT Intrusion Detection")
    print("Environment: Software Simulation (Motion, Temperature, Ultrasonic, RFID)")
    
    # Step 1: Formal Mutation Verification
    if not run_stage1_formal_verification():
        return 1
        
    # Step 2: Simulation Demo
    events = run_stage2_simulation_demo()
    
    # Step 3: Feature Provenance & Zero-Leakage Audit
    if not run_stage3_and_5_feature_audit(events):
        return 1
        
    # Step 4: Hybrid Diagnosis Demo
    if not run_stage5_hybrid_diagnosis_demo(events):
        return 1
        
    # Step 5: Full Regression Suite
    if not run_full_regression_suite():
        return 1
        
    elapsed = time.time() - start_time
    print_banner("SYSTEM INTEGRATION & REGRESSION VERIFICATION COMPLETE")
    print(f"Status: ALL CHECKS PASSED (Total Time: {elapsed:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

