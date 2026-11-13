"""
Enrich the existing packet-feature CSV with:
  - iat_seconds          : per-packet inter-arrival time within each capture
  - rolling_10s_pkt_rate : packets in the trailing 10-second window (int)

Input :  data/base_paper/all_packet_features.csv   (~407 MB, 2.98 M rows)
Output:  data/base_paper/all_packet_features_iat.csv

The script reads in chunks to keep memory usage low (~2 GB peak).
IAT is computed per capture after sorting by timestamp.

Run from any directory:
    .venv\\Scripts\\python.exe data\\base_paper\\add_iat_features.py
"""

from pathlib import Path
import csv
import pandas as pd
import numpy as np

ROOT   = Path(__file__).resolve().parents[2]
INPUT  = ROOT / "data" / "base_paper" / "all_packet_features.csv"
OUTPUT = ROOT / "data" / "base_paper" / "all_packet_features_iat.csv"

WINDOW_SECONDS = 10.0   # rolling-window size for packet rate


def compute_iat_and_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame for a single capture (already sorted by timestamp),
    compute IAT and rolling 10-second packet count per packet.

    IAT for the first packet of each capture is set to 0.0.
    """
    ts = df["timestamp"].values.astype(float)

    # Inter-arrival time: difference from previous packet's timestamp.
    iat = np.empty(len(ts), dtype=float)
    iat[0] = 0.0
    iat[1:] = ts[1:] - ts[:-1]
    # Guard against clock issues (negative IAT -> clamp to 0).
    iat = np.clip(iat, 0.0, None)

    # Rolling 10-second window count (number of packets in (t-10, t]).
    rate = np.empty(len(ts), dtype=int)
    left = 0
    for right in range(len(ts)):
        while ts[right] - ts[left] > WINDOW_SECONDS:
            left += 1
        rate[right] = right - left + 1   # inclusive count

    df = df.copy()
    df["iat_seconds"]           = iat
    df["rolling_10s_pkt_rate"]  = rate
    return df


def main():
    print(f"Input : {INPUT}")
    print(f"Output: {OUTPUT}")

    if not INPUT.exists():
        raise SystemExit(f"ERROR: {INPUT} not found. Run TShark extraction first.")

    CHUNK = 200_000
    pending: dict = {}   # capture_file -> list of DataFrames
    total_in  = 0
    total_out = 0
    writer    = None
    out_fh    = None

    try:
        out_fh = OUTPUT.open("w", encoding="utf-8", newline="")

        for chunk_idx, chunk in enumerate(
            pd.read_csv(INPUT, chunksize=CHUNK, low_memory=False)
        ):
            total_in += len(chunk)
            chunk["timestamp"] = pd.to_numeric(
                chunk["timestamp"], errors="coerce"
            ).fillna(0.0)

            for capture, grp in chunk.groupby("capture_file", sort=False):
                pending.setdefault(capture, []).append(grp)

            # Flush captures that did not appear in this chunk (they are done).
            recent   = set(chunk["capture_file"].unique())
            to_flush = [c for c in list(pending.keys()) if c not in recent]

            for cap in to_flush:
                parts  = pending.pop(cap)
                cap_df = pd.concat(parts, ignore_index=True)
                cap_df = cap_df.sort_values("timestamp", kind="stable")
                cap_df = compute_iat_and_rate(cap_df)

                if writer is None:
                    writer = csv.DictWriter(
                        out_fh,
                        fieldnames=list(cap_df.columns),
                        extrasaction="ignore",
                    )
                    writer.writeheader()

                writer.writerows(cap_df.to_dict("records"))
                total_out += len(cap_df)

            if (chunk_idx + 1) % 5 == 0:
                print(
                    f"  Processed {total_in:,} input rows, "
                    f"written {total_out:,} enriched rows …"
                )

        # Flush remaining captures.
        for cap, parts in pending.items():
            cap_df = pd.concat(parts, ignore_index=True)
            cap_df = cap_df.sort_values("timestamp", kind="stable")
            cap_df = compute_iat_and_rate(cap_df)

            if writer is None:
                writer = csv.DictWriter(
                    out_fh,
                    fieldnames=list(cap_df.columns),
                    extrasaction="ignore",
                )
                writer.writeheader()

            writer.writerows(cap_df.to_dict("records"))
            total_out += len(cap_df)

    finally:
        if out_fh is not None:
            out_fh.close()

    print(f"\nDone.  Input rows: {total_in:,}  Output rows: {total_out:,}")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
