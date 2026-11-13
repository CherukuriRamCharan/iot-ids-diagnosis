"""Run reproducible offline project checks and optionally probe local MQTT."""

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "project_diagnostics.json"
BACKUP_TARGETS = (
    "features/mqtt_traffic_features.csv",
    "features/mqtt_payload_features.csv",
    "results/mqtt_traffic_summary.csv",
    "results/packet_rate_report.csv",
    "results/rule_detection_report.csv",
    "results/combined_diagnosis_report.csv",
    "results/project_diagnostics.json",
)
EXISTING_CSVS = tuple(
    path for path in BACKUP_TARGETS if path.lower().endswith(".csv")
)
SCRIPT_TIMEOUT = 120
REQUIRED_FILES = (
    "README.md",
    "diagnosis_config.py",
    "features/extract_mqtt_traffic.py",
    "features/extract_mqtt_payloads.py",
    "ids/diagnosis_rules.py",
    "ids/evaluate_packet_rates.py",
    "ids/evaluate_rules_on_captures.py",
    "ids/create_combined_diagnosis.py",
    "tests/test_diagnosis_rules.py",
)
PIPELINE = (
    "features/extract_mqtt_traffic.py",
    "features/extract_mqtt_payloads.py",
    "ids/evaluate_packet_rates.py",
    "ids/evaluate_rules_on_captures.py",
    "ids/create_combined_diagnosis.py",
)


def check(name, status, detail, critical=False):
    return {
        "name": name,
        "status": status,
        "detail": str(detail),
        "critical": bool(critical),
    }


def run_command(command, timeout=SCRIPT_TIMEOUT, cwd=ROOT):
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_seconds": round(time.monotonic() - started, 3),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "stdout": _text(exc.stdout),
            "stderr": _text(exc.stderr),
            "duration_seconds": round(time.monotonic() - started, 3),
            "timed_out": True,
            "error": f"Timed out after {timeout} seconds; child process was terminated.",
        }
    except OSError as exc:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "duration_seconds": round(time.monotonic() - started, 3),
            "timed_out": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _integer(value):
    try:
        number = float(value)
        if number >= 0 and number.is_integer():
            return int(number)
    except (TypeError, ValueError, OverflowError):
        pass
    return None


def _numeric(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames or [], list(reader)


def backup_generated_outputs():
    existing = [ROOT / relative for relative in BACKUP_TARGETS
                if (ROOT / relative).is_file()]
    if not existing:
        return {"directory": None, "files": []}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_root = ROOT / "results" / "diagnostic_backups" / stamp
    copied = []
    for source in existing:
        relative = source.relative_to(ROOT)
        destination = backup_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append({
            "source": str(relative),
            "backup": str(destination.relative_to(ROOT)),
            "size_bytes": source.stat().st_size,
        })
    return {"directory": str(backup_root.relative_to(ROOT)), "files": copied}


def check_csv(path, required_columns):
    if not path.is_file():
        return check(f"CSV {path.relative_to(ROOT)}", "fail", "File does not exist.", True), []
    try:
        columns, rows = read_csv(path)
    except (OSError, UnicodeError, csv.Error) as exc:
        return check(f"CSV {path.relative_to(ROOT)}", "fail",
                     f"Could not parse CSV: {exc}", True), []
    missing = sorted(set(required_columns) - set(columns))
    if missing:
        return check(f"CSV {path.relative_to(ROOT)}", "fail",
                     f"Missing columns: {missing}", True), rows
    if rows and any(not (row.get("capture_file") or "").strip() for row in rows):
        return check(f"CSV {path.relative_to(ROOT)}", "fail",
                     "Rows with missing capture_file values.", True), rows
    return check(
        f"CSV {path.relative_to(ROOT)}", "pass",
        f"{len(rows)} rows; required schema present; capture_file populated.",
    ), rows


def classify_capture_pair(previous, current):
    same_stream = bool(
        previous["tcp.stream"]
        and current["tcp.stream"]
        and previous["tcp.stream"] == current["tcp.stream"]
    )
    different_streams = bool(
        previous["tcp.stream"]
        and current["tcp.stream"]
        and previous["tcp.stream"] != current["tcp.stream"]
    )
    same_direction = bool(
        previous["tcp.srcport"]
        and previous["tcp.dstport"]
        and (previous["tcp.srcport"], previous["tcp.dstport"])
        == (current["tcp.srcport"], current["tcp.dstport"])
    )
    if (
        same_stream and same_direction and previous["tcp.seq"]
        and current["tcp.seq"]
    ):
        if (
            previous["tcp.seq"] == current["tcp.seq"]
            and previous["tcp.len"] == current["tcp.len"]
        ):
            return "same_tcp_sequence_segment_or_capture_duplicate"
        return "distinct_tcp_sequence_ranges; likely distinct MQTT PUBLISH records"
    if different_streams:
        return "different_tcp_streams; likely distinct flows"
    return "insufficient_tcp_sequence_evidence"


def capture_duplicate_probe(tshark, capture, timeout=60):
    fields = (
        "frame.number", "frame.time_epoch", "tcp.stream", "tcp.srcport",
        "tcp.dstport", "tcp.seq", "tcp.len", "mqtt.topic", "mqtt.qos",
        "mqtt.dupflag", "mqtt.msg",
    )
    command = [
        tshark, "-r", str(capture), "-Y", "mqtt.msgtype == 3",
        "-T", "fields", "-E", "separator=/t", "-E", "quote=d",
        "-E", "occurrence=f",
    ]
    for field in fields:
        command.extend(("-e", field))
    result = run_command(command, timeout=timeout)
    if result["returncode"] != 0:
        raise RuntimeError(
            result.get("error") or result["stderr"] or
            f"TShark exited {result['returncode']}"
        )

    rows = []
    for line in result["stdout"].splitlines():
        values = next(csv.reader([line], delimiter="\t"))
        values.extend([""] * (len(fields) - len(values)))
        rows.append(dict(zip(fields, values)))

    exact_observation_groups = {}
    repeated_payload_pairs = []
    seen_rows = []
    for row in rows:
        if not row["mqtt.msg"]:
            seen_rows.append(row)
            continue
        payload_key = (row["mqtt.topic"], row["mqtt.msg"])
        exact_key = (
            row["tcp.stream"], row["tcp.srcport"], row["tcp.dstport"],
            row["tcp.seq"], row["tcp.len"], row["mqtt.topic"], row["mqtt.msg"],
        )
        if all(exact_key):
            exact_observation_groups.setdefault(exact_key, []).append(row)
        for previous in reversed(seen_rows):
            if (previous["mqtt.topic"], previous["mqtt.msg"]) != payload_key:
                continue
            try:
                delta = abs(float(row["frame.time_epoch"]) -
                            float(previous["frame.time_epoch"]))
            except (TypeError, ValueError):
                continue
            if delta <= 0.01:
                repeated_payload_pairs.append({
                    "frames": [previous["frame.number"], row["frame.number"]],
                    "delta_seconds": delta,
                    "same_tcp_stream": bool(
                        previous["tcp.stream"]
                        and row["tcp.stream"]
                        and previous["tcp.stream"] == row["tcp.stream"]
                    ),
                    "same_direction_ports": (
                        bool(previous["tcp.srcport"] and previous["tcp.dstport"])
                        and (
                            previous["tcp.srcport"], previous["tcp.dstport"]
                        ) == (row["tcp.srcport"], row["tcp.dstport"])
                    ),
                    "dup_flags": [previous["mqtt.dupflag"], row["mqtt.dupflag"]],
                    "tcp_sequences": [previous["tcp.seq"], row["tcp.seq"]],
                    "interpretation": classify_capture_pair(previous, row),
                })
                break
        seen_rows.append(row)
    exact_groups = [
        {"key": list(key), "frames": [row["frame.number"] for row in group]}
        for key, group in exact_observation_groups.items() if len(group) > 1
    ]
    return {
        "capture": str(capture.relative_to(ROOT)),
        "tshark_returncode": result["returncode"],
        "tshark_duration_seconds": result["duration_seconds"],
        "publish_rows": len(rows),
        "near_simultaneous_identical_payload_pairs": repeated_payload_pairs,
        "same_stream_same_sequence_observation_groups": exact_groups,
        "interpretation": (
            "TShark field evidence is summarized, not treated as proof by "
            "itself. Same stream/direction and identical TCP sequence plus "
            "length/payload supports retransmission or duplicate capture "
            "observation; different TCP sequence numbers support distinct "
            "MQTT publications. Inspect QoS/DUP and raw frames before a "
            "final attribution."
        ),
        "tshark_stderr": result["stderr"],
    }


def mqtt_probe_worker(message_count):
    import threading
    import paho.mqtt.client as mqtt

    host = "localhost"
    port = 1883
    topic = f"diagnostics/{uuid.uuid4().hex}"
    client_suffix = uuid.uuid4().hex[:12]
    connected = threading.Event()
    subscribed = threading.Event()
    deliveries_complete = threading.Event()
    received = Counter()
    received_lock = threading.Lock()

    def client():
        try:
            return mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"ids-diag-{client_suffix}-{uuid.uuid4().hex[:4]}",
            )
        except AttributeError:
            return mqtt.Client(client_id=f"ids-diag-{client_suffix}")

    subscriber = client()
    publisher = client()
    publisher_connected = threading.Event()

    def on_connect(event):
        def callback(*args):
            reason = args[3]
            if not getattr(reason, "is_failure", bool(reason)):
                event.set()
        return callback

    subscriber.on_connect = on_connect(connected)
    publisher.on_connect = on_connect(publisher_connected)
    subscriber.on_subscribe = lambda _c, _u, _mid, _q, _p: subscribed.set()

    def on_message(_client, _userdata, message):
        try:
            item = json.loads(message.payload.decode("utf-8"))
            if item.get("run_id") == client_suffix:
                with received_lock:
                    received[item.get("sequence")] += 1
                    if sum(received.values()) >= message_count:
                        deliveries_complete.set()
        except (UnicodeDecodeError, json.JSONDecodeError):
            return

    subscriber.on_message = on_message
    try:
        subscriber.connect(host, port, keepalive=10)
        subscriber.loop_start()
        if not connected.wait(5):
            raise RuntimeError("Subscriber connection callback timed out.")
        subscriber.subscribe(topic, qos=1)
        if not subscribed.wait(5):
            raise RuntimeError("Subscription acknowledgement timed out.")
        publisher.connect(host, port, keepalive=10)
        publisher.loop_start()
        if not publisher_connected.wait(5):
            raise RuntimeError("Publisher connection callback timed out.")
        for sequence in range(message_count):
            info = publisher.publish(
                topic,
                json.dumps({"run_id": client_suffix, "sequence": sequence}),
                qos=1,
            )
            info.wait_for_publish(timeout=5)
            if not info.is_published():
                raise RuntimeError(f"Publish {sequence} was not acknowledged.")
        deliveries_complete.wait(8)
        with received_lock:
            delivery_count = sum(received.values())
            unique_count = len(received)
            missing = sorted(set(range(message_count)) - set(received))
            duplicate_sequences = {
                sequence: count for sequence, count in received.items()
                if count > 1
            }
        result = {
            "requested_messages": message_count,
            "delivery_count": delivery_count,
            "unique_sequences_received": unique_count,
            "missing_sequences": missing,
            "duplicate_sequences": duplicate_sequences,
            "passed": (
                delivery_count == message_count
                and unique_count == message_count
                and not missing
                and not duplicate_sequences
            ),
            "topic": topic,
        }
        return result
    finally:
        for mqtt_client in (publisher, subscriber):
            try:
                mqtt_client.disconnect()
            finally:
                mqtt_client.loop_stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-mqtt-test", action="store_true",
        help="Run a bounded local broker test (20 unique QoS 1 messages).",
    )
    parser.add_argument(
        "--skip-pipeline", action="store_true",
        help="Validate existing reports without regenerating them.",
    )
    parser.add_argument("--timeout", type=int, default=SCRIPT_TIMEOUT)
    parser.add_argument("--_mqtt-probe-worker", action="store_true",
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be a positive number of seconds.")

    if args._mqtt_probe_worker:
        try:
            result = mqtt_probe_worker(20)
            print(json.dumps(result))
            return 0 if result["passed"] else 1
        except Exception as exc:
            print(json.dumps({"passed": False, "error": f"{type(exc).__name__}: {exc}"}))
            return 1

    checks = []
    subprocesses = []
    report = {
        "repository_root": str(ROOT),
        "started_at_epoch": time.time(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "prefix": sys.prefix,
        },
        "checks": checks,
        "subprocesses": subprocesses,
        "capture_probe": None,
        "live_mqtt_test": None,
        "backups": None,
        "existing_csv_inventory": [],
    }

    for relative in REQUIRED_FILES:
        path = ROOT / relative
        checks.append(check(
            f"required file {relative}",
            "pass" if path.is_file() else "fail",
            "present" if path.is_file() else "missing",
            not path.is_file(),
        ))

    packages = {}
    for module in ("pandas", "paho.mqtt.client"):
        try:
            packages[module] = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            packages[module] = False
    for module, available in packages.items():
        is_optional_paho = module == "paho.mqtt.client"
        status = (
            "pass" if available else
            ("fail" if args.live_mqtt_test else "skipped")
        )
        checks.append(check(
            f"Python package {module}", status,
            "importable" if available else (
                "required only for --live-mqtt-test; not installed"
                if is_optional_paho and not args.live_mqtt_test
                else "not installed in this interpreter"
            ),
            module == "pandas" or (is_optional_paho and args.live_mqtt_test),
        ))
    checks.append(check(
        "Python interpreter",
        "pass" if sys.version_info >= (3, 10) else "fail",
        f"{sys.version.split()[0]} at {sys.executable}",
        sys.version_info < (3, 10),
    ))

    captures = sorted((ROOT / "captures").glob("**/*.pcapng"))
    report["captures"] = [
        {"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size}
        for path in captures
    ]
    checks.append(check(
        "PCAPNG inventory", "pass" if captures else "fail",
        f"{len(captures)} capture(s) found; original captures are read-only.",
        not captures,
    ))
    for category in ("normal", "attack"):
        count = len(list((ROOT / "captures" / category).glob("*.pcapng")))
        checks.append(check(
            f"{category} capture inventory", "pass" if count else "fail",
            f"{count} file(s)", count == 0,
        ))

    for relative in EXISTING_CSVS:
        path = ROOT / relative
        if not path.is_file():
            checks.append(check(
                f"existing CSV {relative}", "skipped",
                "Not present before diagnostics; it may be generated by the pipeline.",
            ))
            continue
        try:
            columns, rows = read_csv(path)
            csv_info = {
                "path": relative,
                "row_count": len(rows),
                "columns": columns,
                "size_bytes": path.stat().st_size,
            }
            report["existing_csv_inventory"].append(csv_info)
            checks.append(check(
                f"existing CSV {relative}", "pass",
                f'{len(rows)} row(s); {len(columns)} column(s); '
                "read before any pipeline regeneration.",
            ))
        except (OSError, UnicodeError, csv.Error) as exc:
            report["existing_csv_inventory"].append({
                "path": relative,
                "error": f"{type(exc).__name__}: {exc}",
            })
            checks.append(check(
                f"existing CSV {relative}", "warning",
                f"Could not parse pre-existing file; it will be backed up "
                f"before regeneration: {type(exc).__name__}: {exc}",
            ))

    tshark = shutil.which("tshark")
    checks.append(check(
        "TShark availability", "pass" if tshark else "fail",
        tshark or "not found on PATH; PCAP parsing and extraction cannot run.",
        not tshark,
    ))

    backup_ok = True
    if not args.skip_pipeline and tshark and packages["pandas"]:
        try:
            report["backups"] = backup_generated_outputs()
            checks.append(check(
                "preserving existing generated CSV outputs", "pass",
                f'{len(report["backups"]["files"])} existing output file(s) '
                "backed up before regeneration."
                + (f' Backup directory: {report["backups"]["directory"]}'
                   if report["backups"]["directory"] else ""),
            ))
        except OSError as exc:
            backup_ok = False
            checks.append(check(
                "preserving existing generated CSV outputs", "fail",
                f"Could not safely back up existing generated files: {exc}",
                True,
            ))
    if not args.skip_pipeline and tshark and packages["pandas"] and backup_ok:
        pipeline_ok = True
        for relative in PIPELINE:
            if not pipeline_ok:
                subprocesses.append({
                    "script": relative, "status": "skipped",
                    "reason": "A prior pipeline stage failed.",
                })
                continue
            command = [sys.executable, str(ROOT / relative)]
            # Extraction itself bounds each TShark invocation at 60 seconds.
            result = run_command(command, timeout=max(args.timeout, 75))
            success = result["returncode"] == 0 and not result["timed_out"]
            result.update({"script": relative, "status": "pass" if success else "fail"})
            subprocesses.append(result)
            pipeline_ok = success
            checks.append(check(
                f"pipeline {relative}", "pass" if success else "fail",
                result.get("stderr") or result.get("stdout") or
                result.get("error") or "completed",
                not success,
            ))
    else:
        if args.skip_pipeline:
            reason = "Pipeline disabled by --skip-pipeline."
        elif not tshark or not packages["pandas"]:
            reason = (
                "Requires TShark and pandas; "
                + ("TShark unavailable. " if not tshark else "")
                + ("pandas unavailable." if not packages["pandas"] else "")
            )
        else:
            reason = "Pipeline skipped because existing-output backup failed."
        for relative in PIPELINE:
            subprocesses.append({
                "script": relative, "status": "skipped", "reason": reason,
            })
        checks.append(check("extraction/report pipeline", "skipped", reason))

    test_command = [sys.executable, "-m", "unittest", "discover",
                    "-s", "tests", "-v"]
    test_result = run_command(test_command, timeout=min(args.timeout, 60))
    test_ok = test_result["returncode"] == 0 and not test_result["timed_out"]
    test_result.update({"script": "synthetic unit tests",
                        "status": "pass" if test_ok else "fail"})
    subprocesses.append(test_result)
    checks.append(check(
        "synthetic unit tests", "pass" if test_ok else "fail",
        (test_result.get("stderr", "") + test_result.get("stdout", "")).strip()
        or test_result.get("error", "completed"),
        not test_ok,
    ))

    traffic_schema = (
        "capture_file", "traffic_label", "mqtt_packets",
        "mqtt_publish_packets", "capture_duration_seconds",
        "publish_duration_seconds",
    )
    payload_schema = (
        "capture_file", "timestamp_epoch", "mqtt_topic", "device_id",
        "temperature", "motion", "sequence", "decode_status",
    )
    rate_schema = (
        "capture_file", "publish_packets", "max_publishes_in_10_seconds",
        "rate_flagged_packets",
    )
    rule_schema = (
        "capture_file", "range_flag", "repeated_payload_evidence",
        "manipulation_flag", "manipulation_pairing_method",
        "manipulation_unmatched_rows",
    )
    combined_schema = (
        "capture_file", "simulated_scenario", "detection_signals",
        "attack_confirmed", "supporting_evidence",
        "uncertainty_and_limitations",
    )
    csv_specs = (
        ("results/mqtt_traffic_summary.csv", traffic_schema),
        ("features/mqtt_payload_features.csv", payload_schema),
        ("results/packet_rate_report.csv", rate_schema),
        ("results/rule_detection_report.csv", rule_schema),
        ("results/combined_diagnosis_report.csv", combined_schema),
    )
    loaded = {}
    schemas_valid = {}
    for relative, schema in csv_specs:
        result, rows = check_csv(ROOT / relative, schema)
        checks.append(result)
        loaded[relative] = rows
        schemas_valid[relative] = result["status"] == "pass"
    for relative in (
        "features/mqtt_payload_features.csv",
        "results/combined_diagnosis_report.csv",
    ):
        rows = loaded.get(relative, [])
        required_nonempty = (
            ("capture_file", "timestamp_epoch", "mqtt_topic", "decode_status")
            if relative.startswith("features/")
            else ("capture_file", "simulated_scenario", "detection_signals",
                  "attack_confirmed")
        )
        missing = {
            column: sum(not (row.get(column) or "").strip() for row in rows)
            for column in required_nonempty
        }
        bad = {column: count for column, count in missing.items() if count}
        checks.append(check(
            f"required values {relative}",
            "fail" if bad else ("pass" if rows else "skipped"),
            json.dumps(bad) if bad else (
                f"All {len(rows)} rows have required values."
                if rows else "No rows available to validate."
            ),
            bool(bad),
        ))

    reports_ready = all(schemas_valid.get(path, False) for path in (
        "results/mqtt_traffic_summary.csv",
        "features/mqtt_payload_features.csv",
        "results/packet_rate_report.csv",
        "results/rule_detection_report.csv",
        "results/combined_diagnosis_report.csv",
    ))
    if reports_ready:
        summary = loaded["results/mqtt_traffic_summary.csv"]
        payload_rows = loaded["features/mqtt_payload_features.csv"]
        rates = loaded["results/packet_rate_report.csv"]
        rule_rows = loaded["results/rule_detection_report.csv"]
        combined = loaded["results/combined_diagnosis_report.csv"]
        capture_names = {str(path.relative_to(ROOT)) for path in captures}
        report_sets = {
            "traffic summary": {row["capture_file"] for row in summary},
            "rate report": {row["capture_file"] for row in rates},
            "payload rule report": {row["capture_file"] for row in rule_rows},
            "combined report": {row["capture_file"] for row in combined},
        }
        all_counts_match = all(names == capture_names for names in report_sets.values())
        single_row_reports = {
            "traffic summary": summary,
            "rate report": rates,
            "combined report": combined,
        }
        duplicate_rows = {
            name: len(rows) - len({row["capture_file"] for row in rows})
            for name, rows in single_row_reports.items()
            if len(rows) != len({row["capture_file"] for row in rows})
        }
        checks.append(check(
            "capture/report count consistency",
            "pass" if all_counts_match and not duplicate_rows else "fail",
            f"captures={len(capture_names)}; report capture counts="
            + json.dumps({name: len(values) for name, values in report_sets.items()})
            + f"; duplicate rows={json.dumps(duplicate_rows)}",
            not all_counts_match or bool(duplicate_rows),
        ))
        publish_by_capture = {
            row["capture_file"]: _integer(row.get("mqtt_publish_packets"))
            for row in summary
        }
        rate_by_capture = {
            row["capture_file"]: _integer(row.get("publish_packets"))
            for row in rates
        }
        payload_by_capture = {}
        for row in payload_rows:
            payload_by_capture[row["capture_file"]] = (
                payload_by_capture.get(row["capture_file"], 0) + 1
            )
        mismatch = {
            name: {"publish_packets": publish_by_capture.get(name),
                   "payload_rows": count}
            for name, count in payload_by_capture.items()
            if publish_by_capture.get(name) != count
            or publish_by_capture.get(name) is None
        }
        rate_mismatch = {
            name: {"packet_publish_count": publish_by_capture.get(name),
                   "timestamped_publish_count": count}
            for name, count in rate_by_capture.items()
            if count is None or publish_by_capture.get(name) is None
            or count > publish_by_capture[name]
        }
        checks.append(check(
            "PUBLISH/payload row consistency",
            "pass" if not mismatch else "fail",
            json.dumps(mismatch) if mismatch else
            "Payload extraction row count equals PUBLISH packet count per capture.",
            bool(mismatch),
        ))
        checks.append(check(
            "packet/rate report consistency",
            "pass" if not rate_mismatch else "fail",
            json.dumps(rate_mismatch) if rate_mismatch else
            "Timestamped PUBLISH count is no greater than extracted PUBLISH count.",
            bool(rate_mismatch),
        ))
        numeric_fields = {
            "traffic summary": (
                summary,
                ("mqtt_packets", "mqtt_publish_packets",
                 "capture_duration_seconds", "publish_duration_seconds"),
            ),
            "rate report": (
                rates,
                ("publish_packets", "duration_seconds",
                 "average_rate_per_second", "max_publishes_in_10_seconds",
                 "rate_flagged_packets", "window_seconds",
                 "rate_threshold_publishes_per_window"),
            ),
        }
        invalid_numeric = {}
        for name, (rows, columns) in numeric_fields.items():
            count_fields = {
                "mqtt_packets", "mqtt_publish_packets", "publish_packets",
                "max_publishes_in_10_seconds", "rate_flagged_packets",
                "rate_threshold_publishes_per_window",
            }
            invalid_count = 0
            for row in rows:
                for column in columns:
                    number = _numeric(row.get(column))
                    if number is None or (
                        column in count_fields
                        and (number < 0 or not number.is_integer())
                    ):
                        invalid_count += 1
            if invalid_count:
                invalid_numeric[name] = invalid_count
        checks.append(check(
            "numeric count fields",
            "fail" if invalid_numeric else "pass",
            json.dumps(invalid_numeric) if invalid_numeric else
            "All packet-count and threshold fields are non-negative integers.",
            bool(invalid_numeric),
        ))
        normal_captures = {
            row["capture_file"] for row in summary
            if (row.get("traffic_label") or "").lower() == "normal"
        }
        normal_rate_values = [
            _integer(row.get("rate_flagged_packets"))
            for row in rates if row["capture_file"] in normal_captures
        ]
        normal_rate_flags = sum(value for value in normal_rate_values if value is not None)
        normal_payload_flags = sum(
            str(row.get("range_flag", "")).lower() == "true"
            or str(row.get("manipulation_flag", "")).lower() == "true"
            for row in rule_rows if row["capture_file"] in normal_captures
        )
        attack_captures = {
            row["capture_file"] for row in summary
            if (row.get("traffic_label") or "").lower() == "attack"
        }
        attack_rate_hits = sum(
            (_integer(row.get("rate_flagged_packets")) or 0) > 0
            for row in rates if row["capture_file"] in attack_captures
        )
        checks.append(check(
            "normal-capture false-positive overlap",
            "pass" if normal_rate_flags == 0 and normal_payload_flags == 0
            and all(value is not None for value in normal_rate_values) else "warning",
            f"normal PUBLISH rate flags={normal_rate_flags}; "
            f"invalid rate counts={sum(value is None for value in normal_rate_values)}; "
            f"normal payload rule flags={normal_payload_flags}; repeated payload "
            f"evidence is intentionally excluded from attack flags; "
            f"attack captures with packet-rate signals={attack_rate_hits}/"
            f"{len(attack_captures)}.",
        ))
    else:
        checks.append(check(
            "cross-report consistency", "skipped",
            "One or more CSV reports failed schema validation; consistency "
            "checks were not run against incomplete reports.",
        ))

    if tshark and captures:
        normal = [path for path in captures if path.parent.name == "normal"]
        if normal:
            try:
                report["capture_probe"] = capture_duplicate_probe(tshark, normal[0])
                checks.append(check(
                    "duplicate publication capture probe", "pass",
                    f'{report["capture_probe"]["publish_rows"]} PUBLISH frames inspected; '
                    f'{len(report["capture_probe"]["near_simultaneous_identical_payload_pairs"])} '
                    "near-identical payload pairs. See capture_probe for TCP/QoS/DUP evidence.",
                ))
            except (RuntimeError, OSError, csv.Error) as exc:
                checks.append(check(
                    "duplicate publication capture probe", "fail",
                    f"{type(exc).__name__}: {exc}", True,
                ))
        else:
            checks.append(check(
                "duplicate publication capture probe", "skipped",
                "No capture exists under captures/normal.",
            ))
    else:
        checks.append(check(
            "duplicate publication capture probe", "skipped",
            "Requires TShark and at least one PCAPNG capture.",
        ))

    if args.live_mqtt_test:
        live = run_command(
            [sys.executable, str(Path(__file__).resolve()),
             "--_mqtt-probe-worker"],
            timeout=30,
        )
        try:
            live_result = json.loads(live["stdout"].strip().splitlines()[-1])
        except (ValueError, IndexError):
            live_result = {
                "passed": False,
                "error": live.get("stderr") or live.get("error")
                or "Worker produced no JSON result.",
            }
        live.update({
            "script": "optional live MQTT delivery worker",
            "status": "pass" if live_result.get("passed") and not live["timed_out"] else "fail",
        })
        subprocesses.append(live)
        report["live_mqtt_test"] = live_result
        live_ok = bool(live_result.get("passed")) and not live["timed_out"]
        checks.append(check(
            "bounded live MQTT delivery test",
            "pass" if live_ok else "fail",
            live_result,
            not live_ok,
        ))
    else:
        checks.append(check(
            "bounded live MQTT delivery test", "skipped",
            "Not requested. Add --live-mqtt-test to publish 20 uniquely numbered "
            "QoS 1 messages to a unique temporary topic on the existing localhost broker.",
        ))

    git = shutil.which("git")
    if git:
        result = run_command([git, "--no-pager", "status", "--short"], timeout=15)
        report["git_status"] = result
        checks.append(check(
            "git status", "pass" if result["returncode"] == 0 else "fail",
            result["stdout"].strip() or result.get("stderr", "").strip() or "clean",
            result["returncode"] != 0,
        ))
    else:
        report["git_status"] = {"status": "skipped", "reason": "git not found on PATH"}
        checks.append(check("git status", "skipped", "git not found on PATH"))

    report["finished_at_epoch"] = time.time()
    report["critical_failures"] = [
        item for item in checks if item["critical"] and item["status"] == "fail"
    ]
    noncritical_failures = [
        item for item in checks if item["status"] == "fail"
    ]
    warnings = [item for item in checks if item["status"] == "warning"]
    report["overall_status"] = (
        "fail" if report["critical_failures"] else
        ("warning" if noncritical_failures or warnings else "pass")
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Project diagnostics:", report["overall_status"].upper())
    for item in checks:
        print(f'[{item["status"].upper()}] {item["name"]}: {item["detail"]}')
    print(f"Machine-readable report: {REPORT}")
    return 1 if report["critical_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
