"""Backward-compatible imports for processed-data utilities."""

from .data import (
    SplitDefinition,
    StudentClipDataset,
    TeacherPPGDataset,
    discover_teacher_subjects,
    make_loader,
    make_student_split,
    make_teacher_split,
)

__all__ = [
    "SplitDefinition",
    "StudentClipDataset",
    "TeacherPPGDataset",
    "discover_teacher_subjects",
    "make_loader",
    "make_student_split",
    "make_teacher_split",
]
