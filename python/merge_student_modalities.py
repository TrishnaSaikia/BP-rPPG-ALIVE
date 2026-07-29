#!/usr/bin/env python3
"""Match prepared PPG clips to rPPG clips using metadata keys.

The rPPG MATLAB pipeline writes rppg_topk.csv, clip_quality.csv, labels.csv
and metadata.csv. The PPG preparation script writes ppg.csv, labels.csv and
metadata.csv. This utility matches rows by dataset, subject_id, video_id and
clip_index, then writes the synchronized ppg.csv into each student subject
folder. Row-order matching is intentionally not used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["dataset", "subject_id", "video_id", "clip_index"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-root", default="processed/student")
    parser.add_argument("--quality", required=True, choices=["HQ", "LQ", "C_23", "C_40", "MSPM"])
    parser.add_argument("--ppg-root", required=True, help="Prepared PPG dataset root containing dataset/subject folders")
    parser.add_argument("--clip-seconds", type=int, default=4)
    parser.add_argument("--label-tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    student_base = Path(args.student_root) / args.quality / f"{args.clip_seconds}_sec"
    ppg_root = Path(args.ppg_root)
    if not student_base.exists():
        raise FileNotFoundError(student_base)

    written = 0
    for subject_dir in sorted(path for path in student_base.iterdir() if path.is_dir()):
        r_meta_path = subject_dir / "metadata.csv"
        if not r_meta_path.exists():
            raise FileNotFoundError(f"Missing rPPG metadata: {r_meta_path}")
        r_meta = pd.read_csv(
            r_meta_path, dtype={"dataset": str, "subject_id": str, "video_id": str}
        )
        for key in KEYS:
            if key not in r_meta.columns:
                raise ValueError(f"{r_meta_path} is missing key column {key}")

        dataset = str(r_meta.iloc[0]["dataset"])
        subject_id = str(r_meta.iloc[0]["subject_id"])
        ppg_dir = ppg_root / dataset / subject_id
        p_meta_path = ppg_dir / "metadata.csv"
        if not p_meta_path.exists():
            raise FileNotFoundError(f"Prepared PPG metadata not found: {p_meta_path}")
        p_meta = pd.read_csv(
            p_meta_path, dtype={"dataset": str, "subject_id": str, "video_id": str}
        )
        ppg = pd.read_csv(ppg_dir / "ppg.csv", header=None)
        p_labels = pd.read_csv(ppg_dir / "labels.csv", header=None)
        if len(p_meta) != len(ppg) or len(ppg) != len(p_labels):
            raise ValueError(f"Prepared PPG row mismatch in {ppg_dir}")
        p_meta = p_meta.copy()
        p_meta["_ppg_row"] = np.arange(len(p_meta))

        merged = r_meta.merge(p_meta[KEYS + ["_ppg_row"]], on=KEYS, how="left", validate="one_to_one")
        if merged["_ppg_row"].isna().any():
            missing = merged.loc[merged["_ppg_row"].isna(), KEYS]
            raise ValueError(f"No matching PPG clips for {subject_dir}:\n{missing.to_string(index=False)}")
        rows = merged["_ppg_row"].astype(int).to_numpy()
        matched_ppg = ppg.iloc[rows].to_numpy()
        matched_labels = p_labels.iloc[rows, :2].to_numpy()

        r_labels = pd.read_csv(subject_dir / "labels.csv", header=None).iloc[:, :2].to_numpy()
        if r_labels.shape != matched_labels.shape or np.max(np.abs(r_labels - matched_labels)) > args.label_tolerance:
            raise ValueError(f"BP labels disagree between PPG and rPPG data for {subject_dir}")
        np.savetxt(subject_dir / "ppg.csv", matched_ppg, delimiter=",", fmt="%.8g")
        written += len(matched_ppg)

    print(f"Matched and wrote {written} synchronized PPG clips for {args.quality}.")


if __name__ == "__main__":
    main()
