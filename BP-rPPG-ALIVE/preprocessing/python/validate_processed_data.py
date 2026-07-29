#!/usr/bin/env python3
"""Validate prepared teacher or student data before training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from alive.data import StudentClipDataset, TeacherPPGDataset, discover_teacher_subjects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="kind", required=True)

    teacher = subparsers.add_parser("teacher")
    teacher.add_argument("--root", default="processed/teacher_ppg/4_sec")
    teacher.add_argument("--clip-samples", type=int, default=120)

    student = subparsers.add_parser("student")
    student.add_argument("--root", default="processed/student")
    student.add_argument("--quality", required=True)
    student.add_argument("--clip-seconds", type=int, default=4)
    student.add_argument("--fps", type=int, default=30)
    student.add_argument("--k-signals", type=int, default=15)
    student.add_argument("--require-ppg", action="store_true")

    args = parser.parse_args()
    if args.kind == "teacher":
        grouped = discover_teacher_subjects(args.root)
        relative = [str(path.relative_to(Path(args.root))) for paths in grouped.values() for path in paths]
        dataset = TeacherPPGDataset(args.root, relative, clip_samples=args.clip_samples)
        print(f"Teacher data valid: {len(relative)} subjects, {len(dataset)} clips")
    else:
        base = Path(args.root) / args.quality / f"{args.clip_seconds}_sec"
        subjects = sorted(path.name for path in base.iterdir() if path.is_dir())
        dataset = StudentClipDataset(
            args.root,
            args.quality,
            subjects,
            args.clip_seconds,
            args.fps,
            args.k_signals,
            require_ppg=args.require_ppg,
        )
        print(f"Student data valid: {len(subjects)} subjects, {len(dataset)} clips")


if __name__ == "__main__":
    main()
