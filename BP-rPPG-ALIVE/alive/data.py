from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def _read_numeric_csv(path: Path) -> np.ndarray:
    try:
        values = pd.read_csv(path, header=None).to_numpy(dtype=np.float32)
    except Exception as exc:  # pragma: no cover - error includes source path
        raise ValueError(f"Failed to read numeric CSV {path}: {exc}") from exc
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D CSV matrix at {path}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite values found in {path}")
    return values


def _read_metadata(path: Path, n_rows: int, fallback: dict[str, str]) -> pd.DataFrame:
    if path.exists():
        metadata = pd.read_csv(
            path, dtype={"dataset": str, "subject_id": str, "video_id": str}
        )
        if len(metadata) != n_rows:
            raise ValueError(
                f"Row-count mismatch: {path} has {len(metadata)} rows; expected {n_rows}"
            )
    else:
        metadata = pd.DataFrame(index=range(n_rows))

    for key, value in fallback.items():
        if key not in metadata.columns:
            metadata[key] = value
    if "clip_index" not in metadata.columns:
        metadata["clip_index"] = np.arange(n_rows, dtype=int)
    if "video_id" not in metadata.columns:
        metadata["video_id"] = fallback.get("subject_id", "recording")

    required = ["dataset", "subject_id", "video_id", "clip_index"]
    for column in required:
        if metadata[column].isna().any():
            raise ValueError(f"Missing metadata values in column '{column}' at {path}")
    return metadata


@dataclass(frozen=True)
class SplitDefinition:
    train: list[str]
    validation: list[str]
    test: list[str]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SplitDefinition":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            train=list(payload["train"]),
            validation=list(payload["validation"]),
            test=list(payload["test"]),
        )


def _ratio_counts(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    if n <= 0:
        return 0, 0
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    if n >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
    elif n == 2:
        n_train, n_val = 1, 0
    else:
        n_train, n_val = 1, 0
    return n_train, n_val


def make_student_split(
    data_root: str | Path,
    quality: str,
    clip_seconds: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> SplitDefinition:
    base = Path(data_root) / quality / f"{clip_seconds}_sec"
    if not base.exists():
        raise FileNotFoundError(f"Student data directory not found: {base}")
    subjects = sorted(path.name for path in base.iterdir() if path.is_dir())
    if not subjects:
        raise RuntimeError(f"No subject directories found under {base}")
    random.Random(seed).shuffle(subjects)
    n_train, n_val = _ratio_counts(len(subjects), train_ratio, val_ratio)
    return SplitDefinition(
        train=subjects[:n_train],
        validation=subjects[n_train : n_train + n_val],
        test=subjects[n_train + n_val :],
    )


def discover_teacher_subjects(data_root: str | Path) -> dict[str, list[Path]]:
    """Return dataset -> subject directories containing ppg.csv and labels.csv."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Teacher PPG data directory not found: {root}")
    grouped: dict[str, list[Path]] = {}
    for ppg_path in sorted(root.rglob("ppg.csv")):
        subject_dir = ppg_path.parent
        if not (subject_dir / "labels.csv").exists():
            continue
        relative = subject_dir.relative_to(root)
        dataset = relative.parts[0] if len(relative.parts) >= 2 else "dataset"
        grouped.setdefault(dataset, []).append(subject_dir)
    if not grouped:
        raise RuntimeError(f"No teacher subject directories with ppg.csv/labels.csv under {root}")
    return grouped


def make_teacher_split(
    data_root: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> SplitDefinition:
    grouped = discover_teacher_subjects(data_root)
    rng = random.Random(seed)
    train: list[str] = []
    validation: list[str] = []
    test: list[str] = []
    root = Path(data_root)

    for dataset, directories in sorted(grouped.items()):
        directories = list(directories)
        rng.shuffle(directories)
        n_train, n_val = _ratio_counts(len(directories), train_ratio, val_ratio)
        relative = [str(directory.relative_to(root)) for directory in directories]
        train.extend(relative[:n_train])
        validation.extend(relative[n_train : n_train + n_val])
        test.extend(relative[n_train + n_val :])
    return SplitDefinition(train=train, validation=validation, test=test)


class TeacherPPGDataset(Dataset):
    """Combined 4-second PPG clips from one or more datasets."""

    def __init__(
        self,
        data_root: str | Path,
        subject_paths: Iterable[str],
        clip_samples: int = 120,
    ) -> None:
        self.root = Path(data_root)
        self.clip_samples = clip_samples
        self.samples: list[dict] = []

        for relative in subject_paths:
            subject_dir = self.root / relative
            ppg = _read_numeric_csv(subject_dir / "ppg.csv")
            labels = _read_numeric_csv(subject_dir / "labels.csv")
            if ppg.shape[1] != clip_samples:
                raise ValueError(
                    f"{subject_dir / 'ppg.csv'} has {ppg.shape[1]} samples per clip; "
                    f"expected {clip_samples}"
                )
            if labels.shape[1] != 2:
                raise ValueError(f"{subject_dir / 'labels.csv'} must contain exactly SBP,DBP")
            if len(ppg) != len(labels):
                raise ValueError(f"PPG/label row mismatch in {subject_dir}")

            parts = Path(relative).parts
            fallback = {
                "dataset": parts[0] if len(parts) >= 2 else "dataset",
                "subject_id": parts[-1],
            }
            metadata = _read_metadata(subject_dir / "metadata.csv", len(ppg), fallback)
            for index in range(len(ppg)):
                meta = metadata.iloc[index]
                self.samples.append(
                    {
                        "ppg": ppg[index],
                        "bp": labels[index],
                        "dataset": str(meta["dataset"]),
                        "subject_id": str(meta["subject_id"]),
                        "video_id": str(meta["video_id"]),
                        "clip_index": int(meta["clip_index"]),
                    }
                )

        if not self.samples:
            raise RuntimeError("No teacher PPG clips were indexed")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        return {
            "ppg": torch.as_tensor(sample["ppg"], dtype=torch.float32).view(1, -1),
            "bp": torch.as_tensor(sample["bp"], dtype=torch.float32),
            "dataset": sample["dataset"],
            "subject_id": sample["subject_id"],
            "video_id": sample["video_id"],
            "clip_index": sample["clip_index"],
        }


class StudentClipDataset(Dataset):
    """Processed ALIVE clip dataset for student training or testing."""

    def __init__(
        self,
        data_root: str | Path,
        quality: str,
        subjects: Iterable[str],
        clip_seconds: int = 4,
        fps: int = 30,
        k_signals: int = 15,
        require_ppg: bool = False,
    ) -> None:
        self.base = Path(data_root) / quality / f"{clip_seconds}_sec"
        self.clip_samples = clip_seconds * fps
        self.k_signals = k_signals
        self.require_ppg = require_ppg
        self.samples: list[dict] = []

        for subject in subjects:
            subject_dir = self.base / subject
            rppg = _read_numeric_csv(subject_dir / "rppg_topk.csv")
            labels = _read_numeric_csv(subject_dir / "labels.csv")
            qualities = _read_numeric_csv(subject_dir / "clip_quality.csv")
            ppg_path = subject_dir / "ppg.csv"
            ppg = _read_numeric_csv(ppg_path) if ppg_path.exists() else None

            expected_rppg = k_signals * self.clip_samples
            if rppg.shape[1] != expected_rppg:
                raise ValueError(
                    f"{subject_dir / 'rppg_topk.csv'} has {rppg.shape[1]} columns; "
                    f"expected K*L={k_signals}*{self.clip_samples}={expected_rppg}"
                )
            if labels.shape[1] != 2:
                raise ValueError(f"{subject_dir / 'labels.csv'} must contain exactly SBP,DBP")
            if qualities.shape[1] != 1:
                raise ValueError(f"{subject_dir / 'clip_quality.csv'} must contain one value per clip")
            if require_ppg and ppg is None:
                raise FileNotFoundError(
                    f"Student training requires synchronized PPG: missing {ppg_path}"
                )
            if ppg is not None and ppg.shape[1] != self.clip_samples:
                raise ValueError(
                    f"{ppg_path} has {ppg.shape[1]} samples per clip; expected {self.clip_samples}"
                )

            counts = {
                "rppg": len(rppg),
                "labels": len(labels),
                "clip_quality": len(qualities),
            }
            if ppg is not None:
                counts["ppg"] = len(ppg)
            if len(set(counts.values())) != 1:
                raise ValueError(f"Clip-count mismatch in {subject_dir}: {counts}")

            metadata = _read_metadata(
                subject_dir / "metadata.csv",
                len(rppg),
                {"dataset": "BP-rPPG", "subject_id": str(subject)},
            )

            for index in range(len(rppg)):
                meta = metadata.iloc[index]
                dataset = str(meta["dataset"])
                subject_id = str(meta["subject_id"])
                video_id = str(meta["video_id"])
                self.samples.append(
                    {
                        "rppg": rppg[index],
                        "ppg": None if ppg is None else ppg[index],
                        "bp": labels[index],
                        "quality": float(qualities[index, 0]),
                        "dataset": dataset,
                        "subject_id": subject_id,
                        "video_id": video_id,
                        "video_key": f"{dataset}::{subject_id}::{video_id}",
                        "clip_index": int(meta["clip_index"]),
                    }
                )

        if not self.samples:
            raise RuntimeError(f"No student clips found under {self.base}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        output = {
            "rppg": torch.as_tensor(sample["rppg"], dtype=torch.float32).view(
                self.k_signals, self.clip_samples
            ),
            "bp": torch.as_tensor(sample["bp"], dtype=torch.float32),
            "quality": torch.tensor(sample["quality"], dtype=torch.float32),
            "dataset": sample["dataset"],
            "subject_id": sample["subject_id"],
            "video_id": sample["video_id"],
            "video_key": sample["video_key"],
            "clip_index": sample["clip_index"],
        }
        if sample["ppg"] is not None:
            output["ppg"] = torch.as_tensor(sample["ppg"], dtype=torch.float32).view(1, -1)
        return output


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
