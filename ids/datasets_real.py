"""Leakage-safe dataset loaders, train-only masked normalizer, and sequence builder
for Phase 3 (MQTT-UAD primary dataset and MQTTset secondary benchmark).

Enforces all Phase 2.5 independent review conditions:
1. Strict separation between predictive features (X) and metadata/leakage columns.
2. Train-only masked z-score normalization for -1.0 sentinels while preserving
   binary availability indicators (has_mqtt, has_mqtt_topic, has_mqtt_msg, etc.).
3. Strictly within-split, within-file sliding-window sequence generation for MQTT-UAD
   (zero boundary-straddling windows).
4. Complete separation between MQTT-UAD and MQTTset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UAD_DIR = ROOT / "datasets" / "processed" / "mqtt_uad"
DEFAULT_MQTTSET_DIR = ROOT / "datasets" / "processed" / "mqttset"

MQTT_UAD_CLASSES: tuple[str, ...] = ("normal", "DoS", "mitm", "intrusion")
MQTTSET_CLASSES: tuple[str, ...] = (
    "Normal",
    "Flood",
    "MalariaDoS",
    "Bruteforce",
    "Malformed",
    "Slowite",
)

# Binary indicator columns preserved unchanged during normalization
UAD_PRESERVE_BINARY_COLS: frozenset[str] = frozenset(
    {
        "has_ip",
        "has_tcp",
        "is_mqtt_port",
        "has_mqtt",
        "has_mqtt_topic",
        "has_mqtt_msg",
    }
)
MQTTSET_PRESERVE_BINARY_COLS: frozenset[str] = frozenset({"has_inf_rate"})

# Columns that must NEVER enter predictive feature matrix X
FORBIDDEN_LEAKAGE_COLS: frozenset[str] = frozenset(
    {
        # Split / target / traceability columns
        "source_file",
        "source_row_idx",
        "split",
        "type",
        "label",
        "Cat",
        # MQTT-UAD capture / time / host / string leakage columns
        "frame.number",
        "frame.time_epoch",
        "frame.time_relative",
        "ip.src",
        "ip.dst",
        "eth.src",
        "eth.dst",
        "tcp.srcport",
        "tcp.dstport",
        "mqtt.topic",
        "mqtt.clientid",
        "mqtt.msg",
        "mqtt.protoname",
        "mqtt.hdrflags",
        "mqtt.conflags",
        "mqtt.conack.flags",
        # MQTTset shortcut / identifier leakage columns
        "Flow_ID",
        "Src_IP",
        "Dst_IP",
        "Src_Port",
        "Dst_Port",
        "Protocol",
        "Timestamp",
    }
)


@dataclass
class SplitDataset:
    """Container for train/val/test splits of a single dataset."""

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    train_groups: np.ndarray
    val_groups: np.ndarray
    test_groups: np.ndarray
    train_row_ids: np.ndarray
    val_row_ids: np.ndarray
    test_row_ids: np.ndarray
    metadata: pd.DataFrame | None = None


def assert_no_leakage_columns(feature_names: Iterable[str]) -> None:
    """Raises ValueError if any forbidden metadata or leakage column is in features."""
    bad = sorted(set(feature_names) & FORBIDDEN_LEAKAGE_COLS)
    if bad:
        raise ValueError(f"Forbidden metadata/leakage columns in feature matrix: {bad}")


class MaskedStandardScaler:
    """Train-only z-score normalizer that masks sentinel values (e.g., -1.0).

    - Fits mean and std exclusively on non-sentinel, finite entries of X_train.
    - Preserves specified binary availability indicators unchanged.
    - Handles all-missing or constant columns safely by mapping them to 0.0.
    - Maps sentinel entries to 0.0 after normalization.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        sentinel_value: float | None = -1.0,
        preserve_cols: frozenset[str] = frozenset(),
    ) -> None:
        self.feature_names = tuple(feature_names)
        self.sentinel_value = sentinel_value
        self.preserve_cols = preserve_cols
        self.preserve_mask = np.array(
            [c in preserve_cols for c in self.feature_names], dtype=bool
        )
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.constant_or_empty_: np.ndarray | None = None

    def fit(self, X_train: np.ndarray) -> "MaskedStandardScaler":
        X = np.asarray(X_train, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected 2D array with {len(self.feature_names)} columns, got {X.shape}"
            )

        n_feat = X.shape[1]
        mean = np.zeros(n_feat, dtype=np.float64)
        scale = np.ones(n_feat, dtype=np.float64)
        const_or_empty = np.zeros(n_feat, dtype=bool)

        for j in range(n_feat):
            if self.preserve_mask[j]:
                continue
            col = X[:, j]
            valid = np.isfinite(col)
            if self.sentinel_value is not None:
                valid &= col != self.sentinel_value
            if not np.any(valid):
                const_or_empty[j] = True
                continue
            vals = col[valid]
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=0))
            if s < 1e-12:
                const_or_empty[j] = True
                mean[j] = m
                scale[j] = 1.0
            else:
                mean[j] = m
                scale[j] = s

        self.mean_ = mean
        self.scale_ = scale
        self.constant_or_empty_ = const_or_empty
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.constant_or_empty_ is None:
            raise RuntimeError("MaskedStandardScaler must be fitted before calling transform().")
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected 2D array with {len(self.feature_names)} columns, got {arr.shape}"
            )

        out = np.zeros_like(arr, dtype=np.float32)
        for j in range(arr.shape[1]):
            col = arr[:, j]
            if self.preserve_mask[j]:
                out[:, j] = col.astype(np.float32)
                continue
            if self.constant_or_empty_[j]:
                out[:, j] = 0.0
                continue
            valid = np.isfinite(col)
            if self.sentinel_value is not None:
                valid &= col != self.sentinel_value
            scaled = np.zeros(arr.shape[0], dtype=np.float32)
            scaled[valid] = ((col[valid] - self.mean_[j]) / self.scale_[j]).astype(np.float32)
            out[:, j] = scaled
        return out

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)


def compute_class_weights(y_train: np.ndarray, n_classes: int) -> np.ndarray:
    """Computes balanced inverse-frequency class weights from y_train only."""
    counts = np.bincount(y_train.astype(np.int64), minlength=n_classes).astype(np.float64)
    total = float(np.sum(counts))
    weights = np.ones(n_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = (total / (n_classes * counts[nonzero])).astype(np.float32)
    return weights


def _resolve_feature_cols(df: pd.DataFrame, manifest_path: Path | None) -> tuple[str, ...]:
    if manifest_path is not None and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cols = tuple(manifest["feature_columns"])
    else:
        cols = tuple(c for c in df.columns if c not in FORBIDDEN_LEAKAGE_COLS)
    assert_no_leakage_columns(cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest feature columns missing from CSV: {missing}")
    return cols


def load_mqtt_uad(
    data_dir: Path | str = DEFAULT_UAD_DIR,
    *,
    mqtt_only: bool = False,
    binary: bool = False,
    load_metadata: bool = False,
) -> SplitDataset:
    """Loads processed MQTT-UAD features, labels, and chronological splits."""
    base = Path(data_dir)
    df = pd.read_csv(base / "mqtt_uad_features.csv", low_memory=False)
    if mqtt_only:
        df = df.loc[df["has_mqtt"] == 1].reset_index(drop=True)

    feat_cols = _resolve_feature_cols(df, base / "mqtt_uad_manifest.json")

    if binary:
        class_names: tuple[str, ...] = ("normal", "attack")
        y_all = (df["type"].astype(str) != "normal").astype(np.int64).to_numpy()
    else:
        class_names = MQTT_UAD_CLASSES
        label_map = {name: i for i, name in enumerate(class_names)}
        unknown = set(df["type"].astype(str).unique()) - set(label_map)
        if unknown:
            raise ValueError(f"Unexpected MQTT-UAD labels: {sorted(unknown)}")
        y_all = df["type"].astype(str).map(label_map).to_numpy(dtype=np.int64)

    X_all = df[list(feat_cols)].to_numpy(dtype=np.float32)
    groups_all = df["source_file"].astype(str).to_numpy()
    row_ids_all = df["source_row_idx"].to_numpy(dtype=np.int64)
    splits = df["split"].astype(str).to_numpy()

    tr = splits == "train"
    va = splits == "val"
    te = splits == "test"

    meta_df = None
    if load_metadata:
        meta_df = pd.read_csv(base / "mqtt_uad_metadata.csv", low_memory=False)
        if mqtt_only:
            kept = set(zip(df["source_file"].astype(str), df["source_row_idx"].astype(int)))
            mask = [
                (sf, int(sr)) in kept
                for sf, sr in zip(meta_df["source_file"].astype(str), meta_df["source_row_idx"])
            ]
            meta_df = meta_df.loc[mask].reset_index(drop=True)

    return SplitDataset(
        name="mqtt_uad_mqtt_only" if mqtt_only else "mqtt_uad",
        X_train=X_all[tr],
        y_train=y_all[tr],
        X_val=X_all[va],
        y_val=y_all[va],
        X_test=X_all[te],
        y_test=y_all[te],
        feature_names=feat_cols,
        class_names=class_names,
        train_groups=groups_all[tr],
        val_groups=groups_all[va],
        test_groups=groups_all[te],
        train_row_ids=row_ids_all[tr],
        val_row_ids=row_ids_all[va],
        test_row_ids=row_ids_all[te],
        metadata=meta_df,
    )


def load_mqttset(
    data_dir: Path | str = DEFAULT_MQTTSET_DIR,
    *,
    binary: bool = False,
    load_metadata: bool = False,
) -> SplitDataset:
    """Loads processed MQTTset features, labels, and Flow_ID-grouped splits."""
    base = Path(data_dir)
    df = pd.read_csv(base / "mqttset_features.csv", low_memory=False)
    feat_cols = _resolve_feature_cols(df, base / "mqttset_manifest.json")

    if binary:
        class_names: tuple[str, ...] = ("Normal", "Anomaly")
        y_all = (df["label"].astype(str) != "Normal").astype(np.int64).to_numpy()
    else:
        class_names = MQTTSET_CLASSES
        label_map = {name: i for i, name in enumerate(class_names)}
        unknown = set(df["Cat"].astype(str).unique()) - set(label_map)
        if unknown:
            raise ValueError(f"Unexpected MQTTset Cat labels: {sorted(unknown)}")
        y_all = df["Cat"].astype(str).map(label_map).to_numpy(dtype=np.int64)

    X_all = df[list(feat_cols)].to_numpy(dtype=np.float32)
    row_ids_all = df["source_row_idx"].to_numpy(dtype=np.int64)
    splits = df["split"].astype(str).to_numpy()

    meta_df = pd.read_csv(base / "mqttset_metadata.csv", low_memory=False)
    groups_all = meta_df["Flow_ID"].astype(str).to_numpy()

    tr = splits == "train"
    va = splits == "val"
    te = splits == "test"

    return SplitDataset(
        name="mqttset",
        X_train=X_all[tr],
        y_train=y_all[tr],
        X_val=X_all[va],
        y_val=y_all[va],
        X_test=X_all[te],
        y_test=y_all[te],
        feature_names=feat_cols,
        class_names=class_names,
        train_groups=groups_all[tr],
        val_groups=groups_all[va],
        test_groups=groups_all[te],
        train_row_ids=row_ids_all[tr],
        val_row_ids=row_ids_all[va],
        test_row_ids=row_ids_all[te],
        metadata=meta_df if load_metadata else None,
    )


def build_mqtt_uad_sequences(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    row_ids: np.ndarray,
    *,
    window_size: int = 16,
    stride: int = 1,
    label_mode: str = "last",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds sliding windows strictly within each capture file (group) of one split.

    Args:
        X: 2D feature array for a single split (train, val, or test).
        y: 1D integer label array aligned with X.
        groups: 1D capture-file identifier array (e.g., source_file) aligned with X.
        row_ids: 1D source_row_idx array aligned with X.
        window_size: Number of consecutive packets per window (>= 1).
        stride: Step size between consecutive windows (>= 1).
        label_mode:
            - "last": label of the final packet in the window (default).
            - "any_attack": for binary labels (0/1), 1 if any packet in window > 0.

    Returns:
        X_seq (N, window_size, F), y_seq (N,), seq_groups (N,), seq_end_row_ids (N,).
        Short groups with fewer than `window_size` packets yield 0 windows without error.
    """
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be >= 1.")
    if label_mode not in {"last", "any_attack"}:
        raise ValueError(f"Unsupported label_mode: {label_mode}")

    n_feat = X.shape[1] if X.ndim == 2 else 0
    if len(X) == 0:
        return (
            np.empty((0, window_size, n_feat), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=groups.dtype),
            np.empty((0,), dtype=np.int64),
        )

    seq_x: list[np.ndarray] = []
    seq_y: list[int] = []
    seq_g: list[object] = []
    seq_r: list[int] = []

    # Preserve deterministic group order of appearance
    _, first_pos = np.unique(groups, return_index=True)
    ordered_groups = groups[np.sort(first_pos)]

    for grp in ordered_groups:
        idx = np.where(groups == grp)[0]
        # Ensure chronological row order within the group
        idx = idx[np.argsort(row_ids[idx], kind="mergesort")]
        n = len(idx)
        if n < window_size:
            continue
        for start in range(0, n - window_size + 1, stride):
            win_idx = idx[start : start + window_size]
            seq_x.append(X[win_idx])
            if label_mode == "last":
                seq_y.append(int(y[win_idx[-1]]))
            else:
                seq_y.append(int(np.any(y[win_idx] > 0)))
            seq_g.append(grp)
            seq_r.append(int(row_ids[win_idx[-1]]))

    if not seq_x:
        return (
            np.empty((0, window_size, n_feat), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=groups.dtype),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.stack(seq_x, axis=0).astype(np.float32),
        np.asarray(seq_y, dtype=np.int64),
        np.asarray(seq_g),
        np.asarray(seq_r, dtype=np.int64),
    )

