import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from diagnosis_config import MAX_TEMPERATURE_C, MIN_TEMPERATURE_C

CAPTURE_DIRS = [ROOT / "captures" / "normal", ROOT / "captures" / "attack"]
OUTPUT = ROOT / "features" / "mqtt_payload_features.csv"

def extract(path):
    cmd = [
        "tshark", "-r", str(path),
        "-Y", "mqtt.msgtype == 3",
        "-T", "fields",
        "-E", "separator=/t",
        "-E", "quote=d",
        "-E", "occurrence=f",
        "-e", "frame.time_epoch",
        "-e", "mqtt.topic",
        "-e", "mqtt.msg",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=60
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())

    records = []
    for line in result.stdout.splitlines():
        fields = next(csv.reader([line], delimiter="\t"))
        fields += [""] * (3 - len(fields))
        timestamp, topic, hex_payload = fields[:3]

        payload_text = ""
        device_id = ""
        device_type = ""
        temperature = ""
        motion = ""
        sequence = ""
        decode_status = "unavailable"

        try:
            payload_text = bytes.fromhex(hex_payload).decode("utf-8")
            decode_status = "decoded_text"
            try:
                payload = json.loads(payload_text)
                device_id = payload.get("device_id", "")
                device_type = payload.get("device_type", "")
                temperature = payload.get("temperature", "")
                motion = payload.get("motion", "")
                sequence = payload.get("sequence", "")
                decode_status = "decoded_json"
            except (json.JSONDecodeError, AttributeError):
                pass
        except (ValueError, UnicodeDecodeError):
            payload_text = ""

        try:
            temp_value = float(temperature)
            range_status = (
                "out_of_configured_range"
                if temp_value < MIN_TEMPERATURE_C or temp_value > MAX_TEMPERATURE_C
                else "within_configured_range"
            )
        except (ValueError, TypeError):
            range_status = "not_available"

        records.append({
            "capture_file": str(path.relative_to(ROOT)),
            "traffic_label": path.parent.name,
            "timestamp_epoch": timestamp,
            "mqtt_topic": topic,
            "device_id": device_id,
            "device_type": device_type,
            "temperature": temperature,
            "motion": motion,
            "sequence": sequence,
            "range_rule_result": range_status,
            "payload_text": payload_text,
            "decode_status": decode_status,
        })

    return records

def main():
    captures = []
    for folder in CAPTURE_DIRS:
        if folder.exists():
            captures.extend(sorted(folder.glob("*.pcapng")))

    all_records = []
    failures = []
    for path in captures:
        try:
            rows = extract(path)
            all_records.extend(rows)
            print(f"{path.name}: extracted {len(rows)} PUBLISH payloads")
        except RuntimeError as exc:
            print(f"ERROR: {path.name}: {exc}")
            failures.append(path.name)
        except subprocess.TimeoutExpired:
            print(f"ERROR: {path.name}: TShark timed out.")
            failures.append(path.name)

    if failures:
        raise SystemExit(f"Payload extraction failed for: {', '.join(failures)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "capture_file", "traffic_label", "timestamp_epoch",
        "mqtt_topic", "device_id", "device_type", "temperature",
        "motion", "sequence", "range_rule_result", "payload_text",
        "decode_status"
    ]
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nTotal payload records: {len(all_records)}")
    print(f"Output saved to: {OUTPUT}")

if __name__ == "__main__":
    main()
