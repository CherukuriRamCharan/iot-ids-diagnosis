"""Small, testable helpers for interpreting MQTT payload and packet evidence."""

from collections import Counter
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnosis_config import MAX_TEMPERATURE_C, MIN_TEMPERATURE_C


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def rolling_window_counts(timestamps, window_seconds):
    """Count timestamps in each trailing window (t-window_seconds, t]."""
    values = [_number(value) for value in timestamps]
    valid = sorted(
        (value, index) for index, value in enumerate(values)
        if value is not None
    )
    counts = [0] * len(values)
    left = 0
    for right, (timestamp, index) in enumerate(valid):
        while timestamp - valid[left][0] >= window_seconds:
            left += 1
        counts[index] = right - left + 1
    return counts


def rate_exceeded(window_count, threshold):
    return window_count > threshold


def compare_message_pairs(originals, forwarded):
    """Pair by sequence when present, otherwise by order if counts agree."""
    originals = [row for row in originals if isinstance(row, dict)]
    forwarded = [row for row in forwarded if isinstance(row, dict)]
    if not originals and not forwarded:
        return [], "no_candidate_messages", 0

    sequence_pairs = all(
        row.get("sequence") is not None
        and str(row.get("sequence")).strip() not in ("", "<NA>", "nan")
        for row in originals + forwarded
    )
    if sequence_pairs:
        original_by_sequence = {
            str(row["sequence"]): row for row in originals
        }
        forwarded_by_sequence = {
            str(row["sequence"]): row for row in forwarded
        }
        common = sorted(
            original_by_sequence.keys() & forwarded_by_sequence.keys()
        )
        pairs = [
            (original_by_sequence[key], forwarded_by_sequence[key])
            for key in common
        ]
        return pairs, "sequence", len(originals) + len(forwarded) - 2 * len(pairs)

    if len(originals) == len(forwarded):
        originals.sort(key=lambda row: _number(row.get("timestamp_epoch"))
                       if _number(row.get("timestamp_epoch")) is not None
                       else float("inf"))
        forwarded.sort(key=lambda row: _number(row.get("timestamp_epoch"))
                       if _number(row.get("timestamp_epoch")) is not None
                       else float("inf"))
        return list(zip(originals, forwarded)), "capture_order_equal_counts", 0

    return [], "unpaired_unequal_counts", len(originals) + len(forwarded)


def changed_fields(original, forwarded):
    changed = []
    for field in ("temperature", "motion"):
        left = _number(original.get(field))
        right = _number(forwarded.get(field))
        if left is None and right is None:
            continue
        if left is None or right is None or left != right:
            changed.append(field)
    return changed


def payload_evidence(
    rows,
    min_temperature=MIN_TEMPERATURE_C,
    max_temperature=MAX_TEMPERATURE_C,
):
    """Return value evidence; repeated values are evidence only, not an alert."""
    valid_json = [
        row for row in rows
        if row.get("decode_status") == "decoded_json"
    ]
    out_of_range = []
    for row in valid_json:
        temperature = _number(row.get("temperature"))
        if temperature is not None and not min_temperature <= temperature <= max_temperature:
            out_of_range.append(row)

    repeated_counts = Counter(
        row.get("payload_text") for row in valid_json
        if row.get("payload_text")
    )
    repeated_rows = sum(count - 1 for count in repeated_counts.values())
    return {
        "payload_rows": len(rows),
        "decoded_json_rows": len(valid_json),
        "out_of_range_rows": out_of_range,
        "repeated_payload_rows": repeated_rows,
    }


def diagnosis_signals(range_violations, rate_violations, changed_pairs):
    """Return every applicable evidence-based signal in stable order."""
    signals = []
    if range_violations:
        signals.append("temperature_range_violation")
    if rate_violations:
        signals.append("excessive_publish_rate")
    if changed_pairs:
        signals.append("paired_original_forwarded_value_change")
    return signals
