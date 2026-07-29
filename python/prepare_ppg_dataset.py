#!/usr/bin/env python3
"""Prepare a unified 4-second PPG dataset from heterogeneous sources.

The script accepts one manifest row per PPG recording, segments every
recording into non-overlapping 4-second clips, and resamples every clip to a
common sampling frequency. The repository default is 30 Hz, producing 120
samples per clip so that teacher and student feature vectors have the same
length during ALIVE feature alignment.

Required manifest columns
-------------------------
dataset, subject_id, signal_path, sampling_rate, sbp, dbp

Optional manifest columns
-------------------------
video_id, signal_column, signal_key, delimiter, has_header, skip_rows,
start_sample, end_sample

Supported signal files: CSV/TXT, NPY, NPZ and MAT.
"""

from __future__ import annotations

import argparse
import math
import shutil
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample, resample_poly

REQUIRED_COLUMNS = {
    "dataset",
    "subject_id",
    "signal_path",
    "sampling_rate",
    "sbp",
    "dbp",
}


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == ""


def _as_bool(value, default: bool = False) -> bool:
    if _is_missing(value):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_signal(row: pd.Series, manifest_dir: Path) -> np.ndarray:
    path = Path(str(row["signal_path"]))
    if not path.is_absolute():
        path = manifest_dir / path
    if not path.exists():
        raise FileNotFoundError(f"Signal file not found: {path}")

    suffix = path.suffix.lower()
    signal_column = None if _is_missing(row.get("signal_column")) else row.get("signal_column")
    signal_key = None if _is_missing(row.get("signal_key")) else str(row.get("signal_key"))

    if suffix in {".csv", ".txt"}:
        delimiter = "," if _is_missing(row.get("delimiter")) else str(row.get("delimiter"))
        header = 0 if _as_bool(row.get("has_header"), False) else None
        skip_rows = 0 if _is_missing(row.get("skip_rows")) else int(row.get("skip_rows"))
        frame = pd.read_csv(path, sep=delimiter, header=header, skiprows=skip_rows)
        if signal_column is None:
            values = frame.iloc[:, 0].to_numpy()
        elif isinstance(signal_column, str) and not signal_column.isdigit():
            values = frame[signal_column].to_numpy()
        else:
            values = frame.iloc[:, int(signal_column)].to_numpy()
    elif suffix == ".npy":
        array = np.load(path)
        if array.ndim == 1:
            values = array
        else:
            column = 0 if signal_column is None else int(signal_column)
            values = array[:, column]
    elif suffix == ".npz":
        archive = np.load(path)
        if signal_key is None:
            if len(archive.files) != 1:
                raise ValueError(f"NPZ {path} has multiple arrays; set signal_key in the manifest")
            signal_key = archive.files[0]
        values = archive[signal_key]
    elif suffix == ".mat":
        if signal_key is None:
            raise ValueError(f"MAT file {path} requires signal_key in the manifest")
        values = loadmat(path)[signal_key]
    else:
        raise ValueError(f"Unsupported signal file format: {path.suffix}")

    values = np.asarray(values, dtype=np.float64).squeeze()
    if values.ndim != 1:
        raise ValueError(f"Signal in {path} must resolve to one dimension, got {values.shape}")

    start = 0 if _is_missing(row.get("start_sample")) else int(row.get("start_sample"))
    end = len(values) if _is_missing(row.get("end_sample")) else int(row.get("end_sample"))
    return values[start:end]


def _handle_nan(segment: np.ndarray, policy: str) -> np.ndarray | None:
    finite = np.isfinite(segment)
    if finite.all():
        return segment
    if policy == "drop":
        return None
    if policy == "error":
        raise ValueError("Non-finite values found in a PPG clip")
    if policy == "interpolate":
        if finite.sum() < 2:
            return None
        x = np.arange(len(segment))
        return np.interp(x, x[finite], segment[finite])
    raise ValueError(f"Unknown NaN policy: {policy}")


def _resample_exact(segment: np.ndarray, source_fs: float, target_fs: float, target_len: int) -> np.ndarray:
    ratio = Fraction(target_fs / source_fs).limit_denominator(10000)
    converted = resample_poly(segment, ratio.numerator, ratio.denominator)
    if len(converted) != target_len:
        converted = resample(converted, target_len)
    return np.asarray(converted, dtype=np.float64)


def _normalise(segment: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return segment
    if method == "zscore":
        std = float(segment.std())
        return (segment - segment.mean()) / max(std, 1e-8)
    if method == "minmax":
        minimum = float(segment.min())
        maximum = float(segment.max())
        return (segment - minimum) / max(maximum - minimum, 1e-8)
    raise ValueError(f"Unknown normalisation method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="CSV manifest describing source PPG recordings")
    parser.add_argument("--output-root", default="processed/teacher_ppg/4_sec")
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--target-fs", type=float, default=30.0)
    parser.add_argument("--nan-policy", choices=["drop", "interpolate", "error"], default="drop")
    parser.add_argument("--normalisation", choices=["none", "zscore", "minmax"], default="none")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = pd.read_csv(
        manifest_path,
        dtype={"dataset": str, "subject_id": str, "video_id": str, "signal_path": str},
    )
    missing = REQUIRED_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    target_len = int(round(args.clip_seconds * args.target_fs))
    if target_len <= 1:
        raise ValueError("clip_seconds * target_fs must produce at least two samples")

    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_root}. Use --overwrite to rebuild it."
            )
        shutil.rmtree(output_root)

    grouped: dict[tuple[str, str], dict[str, list]] = defaultdict(
        lambda: {"ppg": [], "labels": [], "metadata": []}
    )
    dropped_clips = 0

    for row_number, row in manifest.iterrows():
        dataset = str(row["dataset"])
        subject_id = str(row["subject_id"])
        video_id = (
            str(row["video_id"])
            if "video_id" in manifest.columns and not _is_missing(row.get("video_id"))
            else f"recording_{row_number:04d}"
        )
        source_fs = float(row["sampling_rate"])
        if source_fs <= 0:
            raise ValueError(f"Invalid sampling_rate at manifest row {row_number}: {source_fs}")
        signal = _load_signal(row, manifest_path.parent)

        duration_seconds = len(signal) / source_fs
        n_clips = int(math.floor(duration_seconds / args.clip_seconds))
        for clip_index in range(n_clips):
            start_time = clip_index * args.clip_seconds
            end_time = (clip_index + 1) * args.clip_seconds
            start_sample = int(round(start_time * source_fs))
            end_sample = int(round(end_time * source_fs))
            raw_clip = signal[start_sample:end_sample]
            if len(raw_clip) < 2:
                dropped_clips += 1
                continue
            raw_clip = _handle_nan(raw_clip, args.nan_policy)
            if raw_clip is None:
                dropped_clips += 1
                continue
            clip = _resample_exact(raw_clip, source_fs, args.target_fs, target_len)
            clip = _normalise(clip, args.normalisation)
            if not np.all(np.isfinite(clip)):
                raise ValueError(f"Prepared clip is non-finite for {dataset}/{subject_id}/{video_id}")

            bucket = grouped[(dataset, subject_id)]
            bucket["ppg"].append(clip.astype(np.float32))
            bucket["labels"].append([float(row["sbp"]), float(row["dbp"])])
            bucket["metadata"].append(
                {
                    "dataset": dataset,
                    "subject_id": subject_id,
                    "video_id": video_id,
                    "clip_index": clip_index,
                    "start_time_sec": start_time,
                    "end_time_sec": end_time,
                    "source_sampling_rate": source_fs,
                    "target_sampling_rate": args.target_fs,
                    "source_signal_path": str(row["signal_path"]),
                }
            )

    if not grouped:
        raise RuntimeError("No PPG clips were prepared")

    for (dataset, subject_id), bucket in grouped.items():
        subject_dir = output_root / dataset / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(subject_dir / "ppg.csv", np.vstack(bucket["ppg"]), delimiter=",", fmt="%.8g")
        np.savetxt(subject_dir / "labels.csv", np.asarray(bucket["labels"]), delimiter=",", fmt="%.8g")
        pd.DataFrame(bucket["metadata"]).to_csv(subject_dir / "metadata.csv", index=False)

    print(
        f"Prepared {sum(len(v['ppg']) for v in grouped.values())} clips "
        f"at {args.target_fs:g} Hz ({target_len} samples/clip); dropped {dropped_clips} clips."
    )


if __name__ == "__main__":
    main()
