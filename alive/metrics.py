from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def bp_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError("predictions and targets must have shape [N, 2]")
    errors = predictions - targets
    return {
        "SBP_MAE": float(np.mean(np.abs(errors[:, 0]))),
        "SBP_ME": float(np.mean(errors[:, 0])),
        "SBP_STD": float(np.std(errors[:, 0], ddof=0)),
        "DBP_MAE": float(np.mean(np.abs(errors[:, 1]))),
        "DBP_ME": float(np.mean(errors[:, 1])),
        "DBP_STD": float(np.std(errors[:, 1], ddof=0)),
    }


def quality_weighted_fusion(
    records: Iterable[dict],
    eps: float = 1e-8,
    label_tolerance: float = 1e-4,
) -> list[dict]:
    """Fuse clip-level BP estimates into one estimate per video.

    The clip weight is the mean quality of its selected top-k temporal
    signals, exactly as described in the paper. Records must contain:
    video_key, prediction (length 2), target (length 2), quality.
    """

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record["video_key"])].append(record)

    fused: list[dict] = []
    for video_key, clips in sorted(grouped.items()):
        predictions = np.asarray([clip["prediction"] for clip in clips], dtype=np.float64)
        targets = np.asarray([clip["target"] for clip in clips], dtype=np.float64)
        qualities = np.asarray([clip["quality"] for clip in clips], dtype=np.float64)

        if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(targets)):
            raise ValueError(f"Non-finite prediction or target in video {video_key}")
        if np.max(np.abs(targets - targets[0])) > label_tolerance:
            raise ValueError(f"Clip labels are inconsistent within video {video_key}")

        qualities = np.where(np.isfinite(qualities), qualities, 0.0)
        qualities = np.maximum(qualities, 0.0)
        if qualities.sum() <= eps:
            weights = np.full_like(qualities, 1.0 / len(qualities))
        else:
            weights = qualities / qualities.sum()

        prediction = (predictions * weights[:, None]).sum(axis=0)
        first = clips[0]
        fused.append(
            {
                "video_key": video_key,
                "dataset": first.get("dataset", ""),
                "subject_id": first.get("subject_id", ""),
                "video_id": first.get("video_id", ""),
                "n_clips": len(clips),
                "SBP_pred": float(prediction[0]),
                "DBP_pred": float(prediction[1]),
                "SBP_true": float(targets[0, 0]),
                "DBP_true": float(targets[0, 1]),
            }
        )
    return fused
