"""Phase 2 validation script for processed MQTT-UAD and MQTTset outputs.

Verifies row counts, label distributions, missing/infinite values, duplicate
handling, selected/excluded feature sets, and train/val/test group overlap.
Writes datasets/processed/phase2_validation_summary.txt.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UAD_DIR = ROOT / "datasets" / "processed" / "mqtt_uad"
SET_DIR = ROOT / "datasets" / "processed" / "mqttset"
OUT_SUMMARY = ROOT / "datasets" / "processed" / "phase2_validation_summary.txt"
CHUNK_SIZE = 50_000


def validate_mqtt_uad(lines: list[str]) -> bool:
    lines.append("=" * 80)
    lines.append("1. MQTT-UAD VALIDATION")
    lines.append("=" * 80)

    manifest = json.loads((UAD_DIR / "mqtt_uad_manifest.json").read_text(encoding="utf-8"))
    feat_cols = manifest["feature_columns"]

    total_rows = 0
    nan_count = 0
    inf_count = 0
    label_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}

    for chunk in pd.read_csv(UAD_DIR / "mqtt_uad_features.csv", chunksize=CHUNK_SIZE, low_memory=False):
        total_rows += len(chunk)
        X = chunk[feat_cols]
        nan_count += int(X.isna().sum().sum())
        arr = X.to_numpy(dtype=np.float64, copy=False)
        inf_count += int(np.isinf(arr).sum())

        for k, v in chunk["type"].value_counts().items():
            label_counts[str(k)] = label_counts.get(str(k), 0) + int(v)
        for k, v in chunk["split"].value_counts().items():
            split_counts[str(k)] = split_counts.get(str(k), 0) + int(v)

    # Check chronological ordering / overlap across splits per source_file
    meta_time = pd.read_csv(
        UAD_DIR / "mqtt_uad_metadata.csv",
        usecols=["source_file", "frame.time_epoch", "split"],
        low_memory=False,
    )
    time_overlap_ok = True
    lines.append(f"Total processed rows: {total_rows:,} (expected 286,186)")
    lines.append(f"Feature columns ({len(feat_cols)}): {feat_cols}")
    lines.append(f"Feature NaN count: {nan_count:,}")
    lines.append(f"Feature +/-inf count: {inf_count:,}")
    lines.append(f"Label distribution: {label_counts}")
    lines.append(f"Split distribution: {split_counts}")
    lines.append("Split-level label distribution per file:")
    for k, v in manifest["split_label_distribution"].items():
        lines.append(f"  - {k}: {v}")

    lines.append("Chronological bounds per file & split:")
    for fname, grp in meta_time.groupby("source_file", sort=False):
        bounds = {}
        for sp in ("train", "val", "test"):
            sub = grp[grp["split"] == sp]["frame.time_epoch"]
            bounds[sp] = (float(sub.min()), float(sub.max()))
        if not (bounds["train"][1] <= bounds["val"][0] and bounds["val"][1] <= bounds["test"][0]):
            time_overlap_ok = False
        lines.append(
            f"  - {fname}: train=[{bounds['train'][0]:.3f}, {bounds['train'][1]:.3f}], "
            f"val=[{bounds['val'][0]:.3f}, {bounds['val'][1]:.3f}], "
            f"test=[{bounds['test'][0]:.3f}, {bounds['test'][1]:.3f}]"
        )

    passed = (total_rows == 286_186) and (nan_count == 0) and (inf_count == 0) and time_overlap_ok
    lines.append(f"MQTT-UAD VALIDATION PASSED: {passed}")
    return passed


def validate_mqttset(lines: list[str]) -> bool:
    lines.append("\n" + "=" * 80)
    lines.append("2. MQTTSET VALIDATION")
    lines.append("=" * 80)

    manifest = json.loads((SET_DIR / "mqttset_manifest.json").read_text(encoding="utf-8"))
    feat_cols = manifest["feature_columns"]

    total_rows = 0
    nan_count = 0
    inf_count = 0
    cat_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}

    for chunk in pd.read_csv(SET_DIR / "mqttset_features.csv", chunksize=CHUNK_SIZE, low_memory=False):
        total_rows += len(chunk)
        X = chunk[feat_cols]
        nan_count += int(X.isna().sum().sum())
        arr = X.to_numpy(dtype=np.float64, copy=False)
        inf_count += int(np.isinf(arr).sum())

        for k, v in chunk["Cat"].value_counts().items():
            cat_counts[str(k)] = cat_counts.get(str(k), 0) + int(v)
        for k, v in chunk["split"].value_counts().items():
            split_counts[str(k)] = split_counts.get(str(k), 0) + int(v)

    # Check Flow_ID overlap across splits
    meta_flows = pd.read_csv(
        SET_DIR / "mqttset_metadata.csv",
        usecols=["Flow_ID", "split"],
        low_memory=False,
    )
    train_fids = set(meta_flows.loc[meta_flows["split"] == "train", "Flow_ID"].astype(str))
    val_fids = set(meta_flows.loc[meta_flows["split"] == "val", "Flow_ID"].astype(str))
    test_fids = set(meta_flows.loc[meta_flows["split"] == "test", "Flow_ID"].astype(str))
    tv_overlap = len(train_fids & val_fids)
    tt_overlap = len(train_fids & test_fids)
    vt_overlap = len(val_fids & test_fids)

    lines.append(f"Raw rows: {manifest['raw_rows']:,} -> Processed rows: {total_rows:,} (duplicates dropped: {manifest['duplicates_dropped']:,})")
    lines.append(f"Feature columns ({len(feat_cols)}): {feat_cols[:10]} ... (+{len(feat_cols)-10} more)")
    lines.append(f"Feature NaN count: {nan_count:,}")
    lines.append(f"Feature +/-inf count: {inf_count:,}")
    lines.append(f"Cat distribution (processed): {cat_counts}")
    lines.append(f"Split distribution: {split_counts}")
    lines.append(f"Flow_ID overlap (train&val={tv_overlap}, train&test={tt_overlap}, val&test={vt_overlap})")

    passed = (
        (total_rows == manifest["processed_rows"])
        and (nan_count == 0)
        and (inf_count == 0)
        and (tv_overlap == 0 and tt_overlap == 0 and vt_overlap == 0)
    )
    lines.append(f"MQTTSET VALIDATION PASSED: {passed}")
    return passed


def main() -> None:
    lines: list[str] = []
    ok_uad = validate_mqtt_uad(lines)
    ok_set = validate_mqttset(lines)
    lines.append("\n" + "=" * 80)
    lines.append(f"OVERALL PHASE 2 VALIDATION PASSED: {ok_uad and ok_set}")
    lines.append("=" * 80)
    text = "\n".join(lines) + "\n"
    OUT_SUMMARY.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

