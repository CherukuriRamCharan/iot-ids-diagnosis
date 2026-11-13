
from pathlib import Path
import pandas as pd
import json
import hashlib
from datetime import datetime

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "datasets" / "raw"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

CHUNK_SIZE = 50000

def inspect_csv(path):
    print(f"\nInspecting: {path.name}", flush=True)

    report = []
    report.append(f"## {path.name}")
    report.append(f"- File size: {path.stat().st_size:,} bytes")
    report.append(f"- SHA-256: {hashlib.sha256(path.read_bytes()).hexdigest()}")

    total_rows = 0
    missing = {}
    label_counts = {}
    sample = None
    columns = []
    dtypes = {}
    chunk_duplicate_count = 0
    parse_errors = []

    try:
        reader = pd.read_csv(
            path,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            encoding_errors="replace",
            on_bad_lines="warn"
        )

        for chunk_number, chunk in enumerate(reader, start=1):
            if sample is None:
                sample = chunk.head(5)
                columns = list(chunk.columns)
                dtypes = {str(k): str(v) for k, v in chunk.dtypes.items()}
                missing = {str(c): 0 for c in chunk.columns}

            total_rows += len(chunk)

            for col in chunk.columns:
                missing[str(col)] += int(chunk[col].isna().sum())

            # Count duplicates within each chunk only.
            # This does not detect duplicates across different chunks.
            chunk_duplicate_count += int(chunk.duplicated().sum())

            # Inspect likely label columns.
            for col in chunk.columns:
                name = str(col).lower()
                if any(word in name for word in
                       ["label", "attack", "type", "class", "category"]):
                    values = chunk[col].fillna("<MISSING>").astype(str)
                    counts = values.value_counts()
                    target = label_counts.setdefault(str(col), {})
                    for value, count in counts.items():
                        target[value] = target.get(value, 0) + int(count)

            if chunk_number % 10 == 0:
                print(f"  Processed {total_rows:,} rows...", flush=True)

        report.append(f"- Total data rows: {total_rows:,}")
        report.append(f"- Columns ({len(columns)}):")
        report.extend([f"  - `{c}` ({dtypes.get(c, 'unknown')})"
                       for c in columns])

        report.append("- Missing values by column:")
        report.extend([f"  - `{c}`: {n:,}"
                       for c, n in missing.items()])

        report.append(
            f"- Duplicate rows within individual chunks: "
            f"{chunk_duplicate_count:,}"
        )
        report.append(
            "- Duplicate limitation: duplicates spanning different chunks "
            "were not checked."
        )

        report.append("- Candidate label distributions:")
        if label_counts:
            for col, counts in label_counts.items():
                report.append(f"  - Column `{col}`:")
                report.extend([
                    f"    - {value}: {count:,}"
                    for value, count in sorted(
                        counts.items(), key=lambda x: -x[1]
                    )
                ])
        else:
            report.append("  - No obvious label column identified.")

        report.append("- First five data rows:")
        if sample is not None:
            report.append("```text")
            report.append(sample.to_string(index=False))
            report.append("```")

        print(f"  Completed: {total_rows:,} rows", flush=True)

    except Exception as e:
        report.append(f"- AUDIT ERROR: {type(e).__name__}: {e}")
        print(f"  ERROR: {e}", flush=True)

    return "\n".join(report)


def main():
    print("Starting read-only dataset audit...", flush=True)
    print(f"Project: {ROOT}", flush=True)

    mqtt_uad = RAW / "mqtt_uad"
    mqttset = RAW / "mqttset"

    if not mqtt_uad.exists() or not mqttset.exists():
        print("ERROR: Expected dataset folders are missing.")
        return

    uad_files = [
        mqtt_uad / "DoS.csv",
        mqtt_uad / "MitM.csv",
        mqtt_uad / "Intrusion.csv"
    ]

    mqttset_files = list(mqttset.rglob("MQTTset.csv"))

    uad_report = [
        "# MQTT-UAD Dataset Audit",
        f"Generated: {datetime.now().isoformat()}",
        "Audit method: pandas chunked CSV inspection.",
        "This report describes observed files, not assumed dataset properties."
    ]

    for path in uad_files:
        if path.exists():
            uad_report.append(inspect_csv(path))
        else:
            uad_report.append(f"## Missing file: {path}")

    uad_output = DOCS / "MQTT_UAD_DATASET_AUDIT.md"
    uad_output.write_text("\n\n".join(uad_report), encoding="utf-8")

    mqttset_report = [
        "# MQTTset Dataset Audit",
        f"Generated: {datetime.now().isoformat()}",
        f"Chunk size: {CHUNK_SIZE:,}",
        "Audit method: pandas chunked CSV inspection.",
        "This report describes observed files, not assumed dataset properties."
    ]

    if mqttset_files:
        for path in mqttset_files:
            mqttset_report.append(inspect_csv(path))
    else:
        mqttset_report.append("MQTTset.csv was not found.")

    mqttset_output = DOCS / "MQTTSET_DATASET_AUDIT.md"
    mqttset_output.write_text(
        "\n\n".join(mqttset_report), encoding="utf-8"
    )

    print("\nAUDIT FINISHED", flush=True)
    print(f"Report 1: {uad_output}", flush=True)
    print(f"Report 2: {mqttset_output}", flush=True)
    print("No original datasets or project code were modified.")


if __name__ == "__main__":
    main()
