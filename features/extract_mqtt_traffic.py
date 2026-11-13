from pathlib import Path
import csv
import subprocess

ROOT = Path(__file__).resolve().parent.parent
CAPTURE_DIRS = [ROOT / "captures" / "normal", ROOT / "captures" / "attack"]
FEATURES_FILE = ROOT / "features" / "mqtt_traffic_features.csv"
SUMMARY_FILE = ROOT / "results" / "mqtt_traffic_summary.csv"

FIELDS = [
    "frame.number", "frame.time_epoch", "mqtt.msgtype", "mqtt.topic",
    "ip.src", "ipv6.src", "ip.dst", "ipv6.dst", "frame.len",
    "tcp.stream", "tcp.srcport", "tcp.dstport", "tcp.seq", "tcp.len",
    "mqtt.qos", "mqtt.dupflag",
]

def extract_capture(path):
    cmd = [
        "tshark", "-r", str(path), "-Y", "mqtt", "-T", "fields",
        "-E", "separator=/t", "-E", "quote=d", "-E", "occurrence=f"
    ]
    for field in FIELDS:
        cmd.extend(["-e", field])

    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=60
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())

    rows = []
    for line in result.stdout.splitlines():
        values = next(csv.reader([line], delimiter="\t"))
        values += [""] * (len(FIELDS) - len(values))
        r = dict(zip(FIELDS, values))
        if not r["frame.number"]:
            continue

        rows.append({
            "capture_file": str(path.relative_to(ROOT)),
            "traffic_label": path.parent.name,
            "frame_number": r["frame.number"],
            "timestamp_epoch": r["frame.time_epoch"],
            "mqtt_message_type": r["mqtt.msgtype"],
            "mqtt_topic": r["mqtt.topic"],
            "source_address": r["ip.src"] or r["ipv6.src"],
            "destination_address": r["ip.dst"] or r["ipv6.dst"],
            "frame_length_bytes": r["frame.len"],
            "tcp_stream": r["tcp.stream"],
            "tcp_source_port": r["tcp.srcport"],
            "tcp_destination_port": r["tcp.dstport"],
            "tcp_sequence_number": r["tcp.seq"],
            "tcp_payload_length_bytes": r["tcp.len"],
            "mqtt_qos": r["mqtt.qos"],
            "mqtt_duplicate_flag": r["mqtt.dupflag"],
        })
    return rows

def main():
    captures = []
    for folder in CAPTURE_DIRS:
        if folder.exists():
            captures.extend(sorted(folder.glob("*.pcapng")))

    if not captures:
        raise SystemExit("No PCAPNG captures found.")

    all_rows = []
    summaries = []
    failures = []

    for capture in captures:
        try:
            rows = extract_capture(capture)
        except RuntimeError as exc:
            print(f"ERROR processing {capture.name}: {exc}")
            failures.append(capture.name)
            continue
        except subprocess.TimeoutExpired:
            print(f"ERROR processing {capture.name}: TShark timed out.")
            failures.append(capture.name)
            continue

        all_rows.extend(rows)
        times = []
        for row in rows:
            try:
                times.append(float(row["timestamp_epoch"]))
            except (ValueError, TypeError):
                pass

        publishes = [r for r in rows if r["mqtt_message_type"] == "3"]
        duration = max(times) - min(times) if len(times) > 1 else 0.0
        publish_times = []
        for row in publishes:
            try:
                publish_times.append(float(row["timestamp_epoch"]))
            except (ValueError, TypeError):
                pass
        publish_duration = (
            max(publish_times) - min(publish_times)
            if len(publish_times) > 1 else 0.0
        )

        summaries.append({
            "capture_file": str(capture.relative_to(ROOT)),
            "traffic_label": capture.parent.name,
            "mqtt_packets": len(rows),
            "mqtt_publish_packets": len(publishes),
            "unique_publish_topics": ";".join(sorted({
                r["mqtt_topic"] for r in publishes if r["mqtt_topic"]
            })),
            "capture_duration_seconds": round(duration, 3),
            "publish_duration_seconds": round(publish_duration, 3),
            "publish_packets_per_second": round(
                len(publishes) / publish_duration, 3
            ) if publish_duration > 0 else 0.0,
        })
        print(f"{capture.name}: {len(rows)} MQTT packets, "
              f"{len(publishes)} PUBLISH packets")

    if failures:
        raise SystemExit(f"Traffic extraction failed for: {', '.join(failures)}")

    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)

    feature_columns = [
        "capture_file", "traffic_label", "frame_number", "timestamp_epoch",
        "mqtt_message_type", "mqtt_topic", "source_address",
        "destination_address", "frame_length_bytes", "tcp_stream",
        "tcp_source_port", "tcp_destination_port", "tcp_sequence_number",
        "tcp_payload_length_bytes", "mqtt_qos", "mqtt_duplicate_flag",
    ]
    with FEATURES_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=feature_columns)
        writer.writeheader()
        writer.writerows(all_rows)

    summary_columns = [
        "capture_file", "traffic_label", "mqtt_packets",
        "mqtt_publish_packets", "unique_publish_topics",
        "capture_duration_seconds", "publish_duration_seconds",
        "publish_packets_per_second"
    ]
    with SUMMARY_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\nExtracted records: {len(all_rows)}")
    print(f"Features saved: {FEATURES_FILE}")
    print(f"Summary saved: {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
