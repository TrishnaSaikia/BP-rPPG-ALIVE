#!/usr/bin/env python3
"""Extract MediaPipe FaceMesh landmarks for every video frame.

The output is a numeric CSV with one row per video frame and 936 columns:
x_0,y_0,...,x_467,y_467. Coordinates are stored in pixels. Frames in which
no face is detected are written as NaN rows. A header is omitted by default
so MATLAB ``readmatrix`` reads frame 1 as row 1.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

N_LANDMARKS = 468


def extract(video_path: str | Path, output_path: str | Path, with_header: bool = False) -> int:
    video_path = Path(video_path)
    output_path = Path(output_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Cannot open video: {video_path}")

    rows: list[list[float]] = []
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not result.multi_face_landmarks:
                rows.append([np.nan] * (N_LANDMARKS * 2))
                continue
            coordinates: list[float] = []
            for point in result.multi_face_landmarks[0].landmark:
                coordinates.extend([point.x * width, point.y * height])
            rows.append(coordinates)
    capture.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        if with_header:
            writer.writerow([value for index in range(N_LANDMARKS) for value in (f"x_{index}", f"y_{index}")])
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--with-header", action="store_true")
    args = parser.parse_args()
    count = extract(args.video, args.out, args.with_header)
    print(f"Saved landmarks for {count} frames to {args.out}")


if __name__ == "__main__":
    main()
