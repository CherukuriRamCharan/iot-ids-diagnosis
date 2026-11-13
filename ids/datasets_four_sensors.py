"""Stage 5.1: Four-Sensor Dataset Loader and Sequence Generation Pipeline.

Loads the simulated four-sensor IoT dataset, performs rigorous feature provenance
filtering and anti-leakage validation, fits normalizers strictly on training data,
and constructs session-isolated sequence windows for neural sequence modeling.

Core Principles:
1. Feature Provenance: Excludes artificial injector watermarks ('has_tamper_flag',
   'is_event_flood') and quarantined internal TFSM states ('observed_state',
   'output_symbol'). Uses strictly 14 observable physical/network features.
2. Zero Leakage: Fits MaskedStandardScaler strictly on the training split (sessions 0-13);
   transforms validation (14-18) and test (19-23) splits using training parameters.
3. Strict Isolation: Sequence windows NEVER cross session boundaries or dataset splits.
4. Last-Event Labeling: Window label is determined by the final packet in the sequence.
5. Class Imbalance Awareness: Computes inverse-frequency class weights from training
   sequences to penalize minority misclassification during training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ids.datasets_real import MaskedStandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOUR_SENSORS_CSV = ROOT / "datasets" / "generated" / "four_sensors" / "four_sensors_dataset.csv"
DEFAULT_MANIFEST_PATH = ROOT / "datasets" / "generated" / "four_sensors" / "dataset_manifest.json"

# ----------------------------------------------------------------------
# Target Attack Classes & Mappings (7 Classes)
# ----------------------------------------------------------------------
FOUR_SENSORS_CLASSES: tuple[str, ...] = (
    "Normal",
    "DoS",
    "Replay",
    "MITM",
    "DataFalsification",
    "Delay",
    "IdentitySpoofing",
)
CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(FOUR_SENSORS_CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: name for i, name in enumerate(FOUR_SENSORS_CLASSES)}

FOUR_SENSORS_BINARY_CLASSES: tuple[str, ...] = ("Normal", "Attack")
BINARY_TO_IDX: dict[str, int] = {"Normal": 0, "Attack": 1}
IDX_TO_BINARY: dict[int, str] = {0: "Normal", 1: "Attack"}

# ----------------------------------------------------------------------
# Feature Provenance & Selection (14 Observable Features)
# ----------------------------------------------------------------------
# Features selected after provenance audit: strictly observable network/payload fields
SELECTED_OBSERVABLE_FEATURES: tuple[str, ...] = (
    "inter_arrival",         # Elapsed time since previous global event (seconds)
    "payload_size",          # MQTT payload length in bytes
    "qos",                   # MQTT header QoS level (0 or 1)
    "seq_delta",             # Per-sensor sequence number delta
    "is_sensor_motion",      # One-hot topic indicator: home/sensor/motion1
    "is_sensor_temp",        # One-hot topic indicator: home/sensor/temperature1
    "is_sensor_ultrasonic",  # One-hot topic indicator: distance/ultrasonic1
    "is_sensor_rfid",        # One-hot topic indicator: security/rfid1
    "is_event_publish",      # One-hot event indicator: PUBLISH
    "is_event_alert",        # One-hot event indicator: ALERT
    "is_event_heartbeat",    # One-hot event indicator: HEARTBEAT (PINGREQ/PINGRESP)
    "is_event_connect",      # One-hot event indicator: CONNECT/CONNACK
    "numeric_reading",       # Observable telemetry measurement (-1.0 sentinel if none)
    "has_alert_flag",        # Observable boolean payload flag (alert / obstacle / motion)
)

# Explicitly excluded features with audit justifications
EXCLUDED_FEATURES_WITH_REASON: dict[str, str] = {
    "has_tamper_flag": (
        "LEAKAGE: Derived from attack injector metadata payload['tampered_by'] == 'mitm_proxy'. "
        "A real MITM proxy does not self-identify via payload watermarks; using this creates an artificial cheat code."
    ),
    "is_event_flood": (
        "LEAKAGE: Derived from synthetic event_type == 'FLOOD' assigned exclusively to DoS flood packets. "
        "Real MQTT network headers do not label packets as FLOOD; DoS must be detected via rate, size, and timing."
    ),
    "observed_state": (
        "QUARANTINED: Internal TFSM state representation (e.g., s2, s5, s9), not observable on network wire."
    ),
    "output_symbol": (
        "QUARANTINED: Internal TFSM output symbol (e.g., o2, o6, o11), not observable on network wire."
    ),
    "is_attack": "GROUND_TRUTH: Target binary label.",
    "attack_type": "GROUND_TRUTH: Target multiclass label.",
    "attack_target": "GROUND_TRUTH: Target sensor ID metadata.",
    "fault_location": "GROUND_TRUTH: Formal mutant / fault label metadata.",
    "original_payload": "GROUND_TRUTH: Pre-tamper payload retained by injector.",
    "original_timestamp": "GROUND_TRUTH: Pre-delay timestamp retained by injector.",
}

FORBIDDEN_LEAKAGE_COLS: frozenset[str] = frozenset(EXCLUDED_FEATURES_WITH_REASON.keys())

# Binary columns preserved without z-score standardization in MaskedStandardScaler
FOUR_SENSORS_PRESERVE_BINARY_COLS: frozenset[str] = frozenset({
    "qos",
    "is_sensor_motion",
    "is_sensor_temp",
    "is_sensor_ultrasonic",
    "is_sensor_rfid",
    "is_event_publish",
    "is_event_alert",
    "is_event_heartbeat",
    "is_event_connect",
    "has_alert_flag",
})


def assert_no_leakage_four_sensors(feature_names: Iterable[str]) -> None:
    """Raises ValueError if any forbidden metadata or leakage column is in features."""
    bad = sorted(set(feature_names) & FORBIDDEN_LEAKAGE_COLS)
    if bad:
        reasons = [f"'{col}': {EXCLUDED_FEATURES_WITH_REASON.get(col, 'Forbidden')}" for col in bad]
        raise ValueError(f"Forbidden metadata/leakage columns detected in feature matrix:\n" + "\n".join(reasons))


# ----------------------------------------------------------------------
# Class Weights Computation
# ----------------------------------------------------------------------
def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Computes inverse-frequency class weights strictly from training sequence labels.

    Formula: w_c = N_samples / (num_classes * N_c)
    For any class with zero instances, defaults to weight 1.0.

    Args:
        y: 1D array of integer class labels.
        num_classes: Total number of classes.

    Returns:
        1D float32 array of shape (num_classes,) containing normalized weights.
    """
    total = len(y)
    weights = np.ones(num_classes, dtype=np.float32)
    if total == 0:
        return weights

    for c in range(num_classes):
        cnt = int((y == c).sum())
        if cnt > 0:
            weights[c] = total / (float(num_classes) * float(cnt))
        else:
            weights[c] = 1.0
    return weights.astype(np.float32)


# ----------------------------------------------------------------------
# Dataset Containers
# ----------------------------------------------------------------------
@dataclass
class FourSensorsSplitDataset:
    """Container for tabular train/val/test splits before sequence windowing."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    train_sessions: np.ndarray
    val_sessions: np.ndarray
    test_sessions: np.ndarray
    train_row_ids: np.ndarray
    val_row_ids: np.ndarray
    test_row_ids: np.ndarray
    scaler: MaskedStandardScaler
    raw_df: pd.DataFrame
    binary: bool


@dataclass
class FourSensorsSequences:
    """Container for chronological sequence windows across all splits."""

    X_train_seq: np.ndarray       # Shape (N_tr, window_size, num_features)
    y_train_seq: np.ndarray       # Shape (N_tr,)
    train_sessions: np.ndarray    # Shape (N_tr,) session IDs
    train_end_row_ids: np.ndarray # Shape (N_tr,) original CSV row IDs

    X_val_seq: np.ndarray         # Shape (N_va, window_size, num_features)
    y_val_seq: np.ndarray         # Shape (N_va,)
    val_sessions: np.ndarray
    val_end_row_ids: np.ndarray

    X_test_seq: np.ndarray        # Shape (N_te, window_size, num_features)
    y_test_seq: np.ndarray        # Shape (N_te,)
    test_sessions: np.ndarray
    test_end_row_ids: np.ndarray

    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    class_weights: np.ndarray     # Shape (num_classes,)
    window_size: int
    stride: int
    binary: bool


# ----------------------------------------------------------------------
# Dataset Loader
# ----------------------------------------------------------------------
def load_four_sensors_dataset(
    csv_path: Optional[Path | str] = None,
    *,
    binary: bool = False,
    feature_names: Sequence[str] = SELECTED_OBSERVABLE_FEATURES,
) -> FourSensorsSplitDataset:
    """Loads four-sensor generated dataset, performs strict integrity & leakage checks,
    and applies train-only scaling.

    Args:
        csv_path: Path to four_sensors_dataset.csv. Defaults to canonical generated dataset.
        binary: If True, maps labels to binary (0=Normal, 1=Attack). Else 7-class.
        feature_names: Feature columns to extract. Defaults to 14 audited observable features.

    Returns:
        FourSensorsSplitDataset with train/val/test splits and fitted scaler.
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_FOUR_SENSORS_CSV
    if not path.is_file():
        raise FileNotFoundError(f"Four-sensor dataset CSV not found at: {path}")

    # Enforce anti-leakage compile-time guarantee
    assert_no_leakage_four_sensors(feature_names)

    df = pd.read_csv(path, low_memory=False)

    # 1. Ground-truth schema mapping for categorical unattacked sentinels:
    # In the four-sensor discrete simulation schema, benign/unattacked events
    # (where attack_type == 'Normal' and is_attack == 0) legitimately have
    # attack_target = 'None' and fault_location = 'None'.
    # Pandas read_csv parses the string literal "None" as NaN by default.
    # We restore the explicit canonical 'None' label strictly for genuine Normal events.
    # Note: Missing values in attack events (is_attack == 1) are never filled,
    # guaranteeing that any true missing attack annotation triggers an integrity failure.
    if "attack_target" in df.columns:
        benign_mask = (df["attack_type"].astype(str) == "Normal") | (df["is_attack"] == 0)
        df.loc[benign_mask & df["attack_target"].isna(), "attack_target"] = "None"
    if "fault_location" in df.columns:
        benign_mask = (df["attack_type"].astype(str) == "Normal") | (df["is_attack"] == 0)
        df.loc[benign_mask & df["fault_location"].isna(), "fault_location"] = "None"

    # 2. Integrity check: missing values
    null_counts = df.isnull().sum()
    if null_counts.any():
        bad_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Dataset contains missing (NaN) values: {bad_cols}")

    # 2. Check required structural columns
    required_cols = {"session_id", "split", "attack_type", "is_attack"} | set(feature_names)
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Required columns missing from dataset CSV: {missing_cols}")

    # 3. Label resolution
    if binary:
        class_names = FOUR_SENSORS_BINARY_CLASSES
        y_all = (df["attack_type"].astype(str) != "Normal").astype(np.int64).to_numpy()
    else:
        class_names = FOUR_SENSORS_CLASSES
        unknown = set(df["attack_type"].astype(str).unique()) - set(CLASS_TO_IDX)
        if unknown:
            raise ValueError(f"Unexpected attack classes in dataset: {sorted(unknown)}")
        y_all = df["attack_type"].astype(str).map(CLASS_TO_IDX).to_numpy(dtype=np.int64)

    # 4. Extract observable feature matrix
    X_raw = df[list(feature_names)].to_numpy(dtype=np.float32)

    # 5. Integrity check: infinite values
    if np.isinf(X_raw).any():
        raise ValueError("Feature matrix contains infinite values (inf / -inf).")

    session_ids = df["session_id"].to_numpy(dtype=np.int64)
    row_ids = np.arange(len(df), dtype=np.int64)
    splits = df["split"].astype(str).to_numpy()

    tr_mask = splits == "train"
    va_mask = splits == "val"
    te_mask = splits == "test"

    if tr_mask.sum() == 0 or va_mask.sum() == 0 or te_mask.sum() == 0:
        raise ValueError("Invalid split partition: one or more splits (train/val/test) are empty.")

    # 6. Fit scaler strictly on training split
    scaler = MaskedStandardScaler(
        feature_names=feature_names,
        sentinel_value=-1.0,
        preserve_cols=FOUR_SENSORS_PRESERVE_BINARY_COLS,
    )
    X_train_norm = scaler.fit_transform(X_raw[tr_mask])
    X_val_norm = scaler.transform(X_raw[va_mask])
    X_test_norm = scaler.transform(X_raw[te_mask])

    return FourSensorsSplitDataset(
        X_train=X_train_norm,
        y_train=y_all[tr_mask],
        X_val=X_val_norm,
        y_val=y_all[va_mask],
        X_test=X_test_norm,
        y_test=y_all[te_mask],
        feature_names=tuple(feature_names),
        class_names=class_names,
        train_sessions=session_ids[tr_mask],
        val_sessions=session_ids[va_mask],
        test_sessions=session_ids[te_mask],
        train_row_ids=row_ids[tr_mask],
        val_row_ids=row_ids[va_mask],
        test_row_ids=row_ids[te_mask],
        scaler=scaler,
        raw_df=df,
        binary=binary,
    )


# ----------------------------------------------------------------------
# Sequence Window Generator
# ----------------------------------------------------------------------
def _slice_session_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sessions: np.ndarray,
    row_ids: np.ndarray,
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates chronological sliding windows strictly within session boundaries."""
    if window_size < 1 or stride < 1:
        raise ValueError("window_size and stride must be >= 1.")

    n_feat = X.shape[1] if X.ndim == 2 else 0
    if len(X) == 0:
        return (
            np.empty((0, window_size, n_feat), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    seq_x: List[np.ndarray] = []
    seq_y: List[int] = []
    seq_sessions: List[int] = []
    seq_end_row_ids: List[int] = []

    # Preserve chronological order of sessions as they appear
    _, first_idx = np.unique(sessions, return_index=True)
    ordered_sessions = sessions[np.sort(first_idx)]

    for sess in ordered_sessions:
        idx = np.where(sessions == sess)[0]
        # Guarantee strict chronological row ordering within session
        idx = idx[np.argsort(row_ids[idx], kind="mergesort")]
        n = len(idx)
        if n < window_size:
            continue

        for start in range(0, n - window_size + 1, stride):
            win_idx = idx[start : start + window_size]
            seq_x.append(X[win_idx])
            # Last-event labeling
            seq_y.append(int(y[win_idx[-1]]))
            seq_sessions.append(int(sess))
            seq_end_row_ids.append(int(row_ids[win_idx[-1]]))

    if not seq_x:
        return (
            np.empty((0, window_size, n_feat), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.stack(seq_x, axis=0).astype(np.float32),
        np.asarray(seq_y, dtype=np.int64),
        np.asarray(seq_sessions, dtype=np.int64),
        np.asarray(seq_end_row_ids, dtype=np.int64),
    )


def build_four_sensors_sequences(
    dataset: FourSensorsSplitDataset,
    *,
    window_size: int = 16,
    stride: int = 1,
) -> FourSensorsSequences:
    """Builds sliding sequence windows strictly within each session across splits.

    Guarantees:
    - Zero sequence windows cross session boundaries.
    - Zero sequence windows cross dataset split boundaries.
    - Last-event labeling: label of sequence is target label of its final packet.
    - Inverse-frequency class weights computed exclusively from y_train_seq.

    Args:
        dataset: FourSensorsSplitDataset containing preprocessed splits.
        window_size: Number of consecutive events per window (default 16).
        stride: Stride between consecutive windows (default 1).

    Returns:
        FourSensorsSequences container.
    """
    X_tr_s, y_tr_s, s_tr, r_tr = _slice_session_sequences(
        dataset.X_train, dataset.y_train, dataset.train_sessions, dataset.train_row_ids, window_size, stride
    )
    X_va_s, y_va_s, s_va, r_va = _slice_session_sequences(
        dataset.X_val, dataset.y_val, dataset.val_sessions, dataset.val_row_ids, window_size, stride
    )
    X_te_s, y_te_s, s_te, r_te = _slice_session_sequences(
        dataset.X_test, dataset.y_test, dataset.test_sessions, dataset.test_row_ids, window_size, stride
    )

    class_weights = compute_class_weights(y_tr_s, len(dataset.class_names))

    return FourSensorsSequences(
        X_train_seq=X_tr_s,
        y_train_seq=y_tr_s,
        train_sessions=s_tr,
        train_end_row_ids=r_tr,
        X_val_seq=X_va_s,
        y_val_seq=y_va_s,
        val_sessions=s_va,
        val_end_row_ids=r_va,
        X_test_seq=X_te_s,
        y_test_seq=y_te_s,
        test_sessions=s_te,
        test_end_row_ids=r_te,
        feature_names=dataset.feature_names,
        class_names=dataset.class_names,
        class_weights=class_weights,
        window_size=window_size,
        stride=stride,
        binary=dataset.binary,
    )


# ----------------------------------------------------------------------
# PyTorch DataLoader Utility
# ----------------------------------------------------------------------
def prepare_four_sensors_loaders(
    seq_data: FourSensorsSequences,
    *,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Wraps sequence datasets into PyTorch DataLoaders.

    Args:
        seq_data: FourSensorsSequences instance.
        batch_size: Mini-batch size (default 128).

    Returns:
        Dictionary containing DataLoaders, class weights, and metadata.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(seq_data.X_train_seq),
            torch.from_numpy(seq_data.y_train_seq),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(seq_data.X_val_seq),
            torch.from_numpy(seq_data.y_val_seq),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(seq_data.X_test_seq),
            torch.from_numpy(seq_data.y_test_seq),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_weights": seq_data.class_weights,
        "class_names": seq_data.class_names,
        "feature_names": seq_data.feature_names,
        "num_features": len(seq_data.feature_names),
        "num_classes": len(seq_data.class_names),
        "split_sizes": [len(seq_data.X_train_seq), len(seq_data.X_val_seq), len(seq_data.X_test_seq)],
        "y_val": seq_data.y_val_seq,
        "y_test": seq_data.y_test_seq,
        "window_size": seq_data.window_size,
        "stride": seq_data.stride,
        "binary": seq_data.binary,
    }

