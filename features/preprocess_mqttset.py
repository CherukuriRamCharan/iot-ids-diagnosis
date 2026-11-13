"""Phase 2: Leakage-free chunked preprocessing for MQTTset.csv (secondary benchmark).

Streams datasets/raw/mqttset/MQTTset.csv in 50,000-row chunks, removes shortcut
leakage and constant columns, deduplicates feature vectors while preserving
source_row_idx mapping, handles +/-inf values without train->test leakage, and
assigns deterministic Flow_ID-grouped splits.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
CHUNK_SIZE = 50_000
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "datasets" / "raw" / "mqttset" / "MQTTset.csv"
OUT_DIR = ROOT / "datasets" / "processed" / "mqttset"

LEAKAGE_ID_COLS = [
    "Flow_ID",
    "Src_IP",
    "Dst_IP",
    "Src_Port",
    "Dst_Port",
    "Protocol",
    "Timestamp",
]

CONSTANT_COLS = [
    "Bwd_PSH_Flags",
    "Fwd_URG_Flags",
    "Bwd_URG_Flags",
    "URG_Flag_Cnt",
    "CWE_Flag_Count",
    "ECE_Flag_Cnt",
    "Fwd_Byts/b_Avg",
    "Fwd_Pkts/b_Avg",
    "Fwd_Blk_Rate_Avg",
    "Bwd_Byts/b_Avg",
    "Subflow_Bwd_Pkts",
    "Active_Mean",
    "Active_Std",
    "Active_Max",
    "Active_Min",
]

LABEL_COLS = ["label", "Cat"]


def assign_flow_grouped_splits(df: pd.DataFrame, flow_ids: pd.Series, seed: int = SEED) -> pd.Series:
    """Assigns deterministic 70/15/15 splits grouped by Flow_ID and stratified by Cat.

    Note: Prevents within-Flow_ID leakage across splits, but does NOT create
    independent capture-level splits because each Cat comes from a single capture.
    """
    rng = np.random.default_rng(seed)
    flow_to_split: dict[str, str] = {}

    # Assign each unique Flow_ID to a split stratified by its primary Cat
    flow_cat = (
        pd.DataFrame({"Flow_ID": flow_ids.values, "Cat": df["Cat"].values})
        .drop_duplicates(subset=["Flow_ID"], keep="first")
    )

    for _, grp in flow_cat.groupby("Cat", sort=True):
        fids = grp["Flow_ID"].to_numpy().copy()
        rng.shuffle(fids)
        n = len(fids)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        for fid in fids[:n_train]:
            flow_to_split[fid] = "train"
        for fid in fids[n_train : n_train + n_val]:
            flow_to_split[fid] = "val"
        for fid in fids[n_train + n_val :]:
            flow_to_split[fid] = "test"

    return flow_ids.map(flow_to_split)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    feat_chunks: list[pd.DataFrame] = []
    meta_chunks: list[pd.DataFrame] = []
    seen_hashes: set[bytes] = set()

    raw_row_count = 0
    dedup_dropped = 0
    inf_counts: dict[str, int] = {"Flow_Pkts/s": 0, "Flow_Byts/s": 0}
    row_offset = 0

    for chunk in pd.read_csv(RAW_PATH, chunksize=CHUNK_SIZE, low_memory=False):
        n = len(chunk)
        raw_row_count += n
        source_row_idx = np.arange(row_offset, row_offset + n, dtype=np.int64)
        row_offset += n

        feature_cols = [
            c for c in chunk.columns if c not in set(LEAKAGE_ID_COLS + CONSTANT_COLS + LABEL_COLS)
        ]

        # Count +/-inf before replacing
        has_inf = np.zeros(n, dtype=np.int8)
        for col in ("Flow_Pkts/s", "Flow_Byts/s"):
            arr = pd.to_numeric(chunk[col], errors="coerce").to_numpy(dtype=np.float64)
            is_inf = np.isinf(arr)
            inf_counts[col] += int(is_inf.sum())
            has_inf = np.maximum(has_inf, is_inf.astype(np.int8))

        # Deduplicate on feature_cols + label + Cat
        dedup_subset = chunk[feature_cols + LABEL_COLS].astype(str).agg("|".join, axis=1)
        keep_mask = np.zeros(n, dtype=bool)
        for i, s in enumerate(dedup_subset):
            h = hashlib.blake2b(s.encode("utf-8", errors="replace"), digest_size=16).digest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                keep_mask[i] = True
            else:
                dedup_dropped += 1

        if not keep_mask.any():
            continue

        kept_chunk = chunk.loc[keep_mask].copy()
        kept_idx = source_row_idx[keep_mask]
        kept_inf = has_inf[keep_mask]

        meta = kept_chunk[LEAKAGE_ID_COLS + LABEL_COLS].copy()
        meta.insert(0, "source_row_idx", kept_idx)
        meta_chunks.append(meta)

        feat = kept_chunk[feature_cols].copy()
        for col in feature_cols:
            feat[col] = pd.to_numeric(feat[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        feat["has_inf_rate"] = kept_inf
        feat.insert(0, "source_row_idx", kept_idx)
        feat["label"] = kept_chunk["label"].astype(str)
        feat["Cat"] = kept_chunk["Cat"].astype(str)
        feat_chunks.append(feat)

    features_df = pd.concat(feat_chunks, ignore_index=True)
    metadata_df = pd.concat(meta_chunks, ignore_index=True)

    # Assign Flow_ID-grouped splits
    splits = assign_flow_grouped_splits(features_df, metadata_df["Flow_ID"].astype(str), seed=SEED)
    features_df["split"] = splits.values
    metadata_df["split"] = splits.values

    # Impute NaNs (from +/-inf) using train-split median only (zero train->test leakage)
    numeric_feat_cols = [
        c for c in features_df.columns if c not in {"source_row_idx", "label", "Cat", "split"}
    ]
    train_mask = features_df["split"] == "train"
    imputed_medians: dict[str, float] = {}
    for col in numeric_feat_cols:
        if features_df[col].isna().any():
            med = float(features_df.loc[train_mask, col].median())
            if np.isnan(med):
                med = 0.0
            features_df[col] = features_df[col].fillna(med)
            imputed_medians[col] = med

    features_path = OUT_DIR / "mqttset_features.csv"
    metadata_path = OUT_DIR / "mqttset_metadata.csv"
    manifest_path = OUT_DIR / "mqttset_manifest.json"

    features_df.to_csv(features_path, index=False)
    metadata_df.to_csv(metadata_path, index=False)

    split_cat_dist = (
        features_df.groupby(["split", "Cat"]).size().unstack(fill_value=0).to_dict(orient="index")
    )

    manifest = {
        "seed": SEED,
        "raw_rows": raw_row_count,
        "processed_rows": int(len(features_df)),
        "duplicates_dropped": int(dedup_dropped),
        "feature_columns": numeric_feat_cols,
        "excluded_leakage_id_columns": LEAKAGE_ID_COLS,
        "dropped_constant_columns": CONSTANT_COLS,
        "raw_inf_counts": inf_counts,
        "train_median_imputed": imputed_medians,
        "cat_distribution_processed": features_df["Cat"].value_counts().to_dict(),
        "label_distribution_processed": features_df["label"].value_counts().to_dict(),
        "split_cat_distribution": split_cat_dist,
        "split_note": (
            "Stratified Flow_ID-grouped 70/15/15 split. Prevents within-flow leakage, "
            "but does NOT overcome single-capture-per-Cat dataset artifact."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"MQTTset preprocessing complete: {raw_row_count:,} raw -> {len(features_df):,} processed "
        f"({dedup_dropped:,} duplicates dropped) -> {OUT_DIR}"
    )


if __name__ == "__main__":
    main()

