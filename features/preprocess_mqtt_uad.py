"""Phase 2: Leakage-free preprocessing and temporal block analysis for MQTT-UAD.

Reads datasets/raw/mqtt_uad/{DoS,MitM,Intrusion}.csv in chunks (read-only) and
writes processed feature tables, metadata/application-evidence tables, and split
assignments to datasets/processed/mqtt_uad/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
CHUNK_SIZE = 50_000
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "datasets" / "raw" / "mqtt_uad"
OUT_DIR = ROOT / "datasets" / "processed" / "mqtt_uad"

RAW_FILES = ["DoS.csv", "MitM.csv", "Intrusion.csv"]

# 19 columns that are 100% NaN across all three files + 3 additional 100% NaN in MitM.csv
# (kept uniform across the combined MQTT-UAD schema).
ALL_NAN_COLS = [
    "frame.time_invalid",
    "frame.coloring_rule.name",
    "frame.coloring_rule.string",
    "frame.comment",
    "frame.comment.expert",
    "frame.file_off",
    "frame.incomplete",
    "frame.interface_id",
    "frame.interface_name",
    "frame.link_nr",
    "frame.md5_hash",
    "mqtt.passwd",
    "mqtt.passwd_len",
    "mqtt.username",
    "mqtt.username_len",
    "mqtt.willmsg",
    "mqtt.willmsg_len",
    "mqtt.willtopic",
    "mqtt.willtopic_len",
]

CONSTANT_COLS = [
    "frame.encap_type",
    "frame.ignored",
    "frame.marked",
    "frame.offset_shift",
]

# Retained in metadata.csv for traceability & post-hoc host attribution, excluded from X
METADATA_ID_COLS = [
    "frame.number",
    "frame.time_epoch",
    "frame.time_relative",
    "ip.src",
    "ip.dst",
    "eth.src",
    "eth.dst",
    "tcp.srcport",
    "tcp.dstport",
]

# Retained in metadata.csv as MQTT application-layer evidence, excluded from X
# (raw topic/clientid strings leak single-lab generator names like mqtt-malaria)
MQTT_STRING_EVIDENCE_COLS = [
    "mqtt.topic",
    "mqtt.clientid",
    "mqtt.msg",
    "mqtt.protoname",
    "mqtt.hdrflags",
    "mqtt.conflags",
    "mqtt.conack.flags",
]

# Safe behavioral packet-level features
PACKET_NUMERIC_COLS = [
    "frame.time_delta",
    "frame.time_delta_displayed",
    "frame.cap_len",
    "frame.len",
]

# Safe MQTT numeric/flag features (missingness handled via has_mqtt + sentinel -1)
MQTT_NUMERIC_COLS = [
    "mqtt.msgtype",
    "mqtt.len",
    "mqtt.qos",
    "mqtt.dupflag",
    "mqtt.retain",
    "mqtt.kalive",
    "mqtt.ver",
    "mqtt.proto_len",
    "mqtt.clientid_len",
    "mqtt.topic_len",
    "mqtt.msgid",
    "mqtt.sub.qos",
    "mqtt.suback.qos",
    "mqtt.conack.val",
    "mqtt.conack.flags.reserved",
    "mqtt.conack.flags.sp",
    "mqtt.conflag.cleansess",
    "mqtt.conflag.passwd",
    "mqtt.conflag.qos",
    "mqtt.conflag.reserved",
    "mqtt.conflag.retain",
    "mqtt.conflag.uname",
    "mqtt.conflag.willflag",
]

MISSING_SENTINEL = -1.0


def assign_contiguous_block_splits(df: pd.DataFrame) -> pd.Series:
    """Assigns contiguous chronological splits (70% train / 15% val / 15% test) per file.

    Note: Because each attack class exists in only one capture file, this does NOT
    produce an independent-capture test set; it only prevents random row-level leakage.
    """
    splits = pd.Series("train", index=df.index, dtype="object")
    for _, grp in df.groupby("source_file", sort=False):
        ordered_idx = grp.sort_values(["frame.time_epoch", "source_row_idx"]).index
        n = len(ordered_idx)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        splits.loc[ordered_idx[n_train : n_train + n_val]] = "val"
        splits.loc[ordered_idx[n_train + n_val :]] = "test"
    return splits


def process_file(fname: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fpath = RAW_DIR / fname
    feat_chunks: list[pd.DataFrame] = []
    meta_chunks: list[pd.DataFrame] = []
    row_offset = 0

    for chunk in pd.read_csv(fpath, chunksize=CHUNK_SIZE, low_memory=False):
        n = len(chunk)
        source_row_idx = np.arange(row_offset, row_offset + n, dtype=np.int64)
        row_offset += n

        # Metadata & application-layer evidence table
        meta = chunk[METADATA_ID_COLS + MQTT_STRING_EVIDENCE_COLS + ["type"]].copy()
        meta.insert(0, "source_row_idx", source_row_idx)
        meta.insert(0, "source_file", fname)
        meta_chunks.append(meta)

        # Predictive feature table X
        feat = pd.DataFrame(index=chunk.index)
        feat["source_file"] = fname
        feat["source_row_idx"] = source_row_idx

        for col in PACKET_NUMERIC_COLS:
            vals = pd.to_numeric(chunk[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            feat[col] = vals.fillna(MISSING_SENTINEL).astype(np.float64)

        # Derived protocol-layer indicators (without raw IP/MAC memorization)
        feat["has_ip"] = chunk["ip.src"].notna().astype(np.int8)
        feat["has_tcp"] = chunk["tcp.srcport"].notna().astype(np.int8)
        src_p = pd.to_numeric(chunk["tcp.srcport"], errors="coerce")
        dst_p = pd.to_numeric(chunk["tcp.dstport"], errors="coerce")
        feat["is_mqtt_port"] = ((src_p == 1883) | (dst_p == 1883)).astype(np.int8)
        feat["has_mqtt"] = chunk["mqtt.msgtype"].notna().astype(np.int8)
        feat["has_mqtt_topic"] = chunk["mqtt.topic"].notna().astype(np.int8)
        feat["has_mqtt_msg"] = chunk["mqtt.msg"].notna().astype(np.int8)

        for col in MQTT_NUMERIC_COLS:
            vals = pd.to_numeric(chunk[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            feat[col] = vals.fillna(MISSING_SENTINEL).astype(np.float64)

        feat["type"] = chunk["type"].astype(str)
        feat_chunks.append(feat)

    return pd.concat(feat_chunks, ignore_index=True), pd.concat(meta_chunks, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_feat: list[pd.DataFrame] = []
    all_meta: list[pd.DataFrame] = []
    for fname in RAW_FILES:
        f_df, m_df = process_file(fname)
        all_feat.append(f_df)
        all_meta.append(m_df)

    features_df = pd.concat(all_feat, ignore_index=True)
    metadata_df = pd.concat(all_meta, ignore_index=True)

    # Assign chronological contiguous block splits using metadata timestamps
    splits = assign_contiguous_block_splits(metadata_df)
    features_df["split"] = splits.values
    metadata_df["split"] = splits.values

    # Check label distribution per split
    split_label_dist = (
        features_df.groupby(["source_file", "split", "type"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )
    split_label_serializable = {f"{k[0]}|{k[1]}": v for k, v in split_label_dist.items()}

    features_path = OUT_DIR / "mqtt_uad_features.csv"
    metadata_path = OUT_DIR / "mqtt_uad_metadata.csv"
    manifest_path = OUT_DIR / "mqtt_uad_manifest.json"

    features_df.to_csv(features_path, index=False)
    metadata_df.to_csv(metadata_path, index=False)

    feature_cols = [
        c for c in features_df.columns if c not in {"source_file", "source_row_idx", "type", "split"}
    ]
    manifest = {
        "seed": SEED,
        "total_rows": int(len(features_df)),
        "feature_columns": feature_cols,
        "dropped_all_nan_columns": ALL_NAN_COLS,
        "dropped_constant_columns": CONSTANT_COLS,
        "excluded_identity_leakage_columns": METADATA_ID_COLS,
        "excluded_mqtt_string_columns": MQTT_STRING_EVIDENCE_COLS,
        "missing_sentinel": MISSING_SENTINEL,
        "label_distribution": features_df["type"].value_counts().to_dict(),
        "split_label_distribution": split_label_serializable,
        "split_note": (
            "Contiguous chronological 70/15/15 block split per file. "
            "Independent capture-level test split is NOT possible because each attack class "
            "appears in only one capture file."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"MQTT-UAD preprocessing complete: {len(features_df):,} rows -> {OUT_DIR}")


if __name__ == "__main__":
    main()

