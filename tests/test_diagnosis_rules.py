import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ids"))

from diagnosis_rules import (  # noqa: E402
    changed_fields,
    compare_message_pairs,
    diagnosis_signals,
    payload_evidence,
    rate_exceeded,
    rolling_window_counts,
)
from tools.run_project_diagnostics import classify_capture_pair  # noqa: E402


class DiagnosisRuleTests(unittest.TestCase):
    def test_repeated_normal_sensor_reading_is_not_an_attack_signal(self):
        evidence = payload_evidence([
            {"decode_status": "decoded_json", "temperature": 22.0,
             "payload_text": '{"temperature":22}'},
            {"decode_status": "decoded_json", "temperature": 22.0,
             "payload_text": '{"temperature":22}'},
        ])
        self.assertEqual(evidence["repeated_payload_rows"], 1)
        self.assertEqual(evidence["out_of_range_rows"], [])
        self.assertEqual(diagnosis_signals(0, 0, 0), [])

    def test_out_of_range_temperature_uses_inclusive_configured_limits(self):
        evidence = payload_evidence([
            {"decode_status": "decoded_json", "temperature": -10},
            {"decode_status": "decoded_json", "temperature": 60},
            {"decode_status": "decoded_json", "temperature": 60.1},
        ], min_temperature=-10, max_temperature=60)
        self.assertEqual(len(evidence["out_of_range_rows"]), 1)

    def test_trailing_window_counts_and_detects_excess_rate(self):
        counts = rolling_window_counts(
            [0, 1, 2, 9.999, 10, 10.001], window_seconds=10
        )
        self.assertEqual(counts, [1, 2, 3, 4, 4, 5])
        self.assertTrue(rate_exceeded(6, 5))
        self.assertFalse(rate_exceeded(5, 5))
        self.assertIn("excessive_publish_rate", diagnosis_signals(0, 1, 0))

    def test_original_forwarded_pairing_prefers_sequence(self):
        pairs, method, unmatched = compare_message_pairs(
            [{"sequence": 2, "temperature": 10},
             {"sequence": 1, "temperature": 20, "motion": 0}],
            [{"sequence": 1, "temperature": 21, "motion": 1},
             {"sequence": 2, "temperature": 10}],
        )
        self.assertEqual(method, "sequence")
        self.assertEqual(unmatched, 0)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(changed_fields(*pairs[0]), ["temperature", "motion"])

    def test_order_pairing_requires_equal_message_counts(self):
        pairs, method, unmatched = compare_message_pairs(
            [{"temperature": 10, "timestamp_epoch": 1},
             {"temperature": 20, "timestamp_epoch": 2}],
            [{"temperature": 11, "timestamp_epoch": 1.1},
             {"temperature": 20, "timestamp_epoch": 2.1}],
        )
        self.assertEqual(method, "capture_order_equal_counts")
        self.assertEqual(unmatched, 0)
        self.assertEqual(len(pairs), 2)
        pairs, method, unmatched = compare_message_pairs(
            [{"temperature": 10}], [{"temperature": 11},
                                     {"temperature": 12}]
        )
        self.assertEqual(method, "unpaired_unequal_counts")
        self.assertEqual(unmatched, 3)

    def test_malformed_and_missing_payload_data_is_safe(self):
        evidence = payload_evidence([
            {"decode_status": "decode_error", "temperature": "bad"},
            {"decode_status": "decoded_json", "temperature": None,
             "payload_text": ""},
        ])
        self.assertEqual(evidence["decoded_json_rows"], 1)
        self.assertEqual(evidence["out_of_range_rows"], [])
        pairs, method, unmatched = compare_message_pairs(
            [{"temperature": None}], []
        )
        self.assertEqual((pairs, method, unmatched),
                         ([], "unpaired_unequal_counts", 1))
        self.assertEqual(changed_fields({}, {}), [])

    def test_capture_pair_needs_tcp_sequence_evidence(self):
        first = {
            "tcp.stream": "1", "tcp.srcport": "1000", "tcp.dstport": "1883",
            "tcp.seq": "500", "tcp.len": "40",
        }
        same_segment = dict(first)
        distinct_segment = {**first, "tcp.seq": "540"}
        self.assertEqual(
            classify_capture_pair(first, same_segment),
            "same_tcp_sequence_segment_or_capture_duplicate",
        )
        self.assertIn(
            "distinct_tcp_sequence",
            classify_capture_pair(first, distinct_segment),
        )
        self.assertEqual(
            classify_capture_pair(
                {**first, "tcp.seq": ""}, {**first, "tcp.seq": ""}
            ),
            "insufficient_tcp_sequence_evidence",
        )

    def test_combined_diagnosis_preserves_all_applicable_signals(self):
        self.assertEqual(
            diagnosis_signals(2, 3, 1),
            [
                "temperature_range_violation",
                "excessive_publish_rate",
                "paired_original_forwarded_value_change",
            ],
        )


if __name__ == "__main__":
    unittest.main()
