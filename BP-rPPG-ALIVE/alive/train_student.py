from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from .config import get, load_config
from .data import (
    SplitDefinition,
    StudentClipDataset,
    make_loader,
    make_student_split,
)
from .losses import DataFidelityLoss, NegativePearsonLoss
from .metrics import bp_metrics, quality_weighted_fusion
from .models import StudentRPPGNetwork, TeacherPPGNetwork
from .utils import load_checkpoint, save_json, set_seed


def _student_kwargs(config: dict) -> dict:
    return {
        "k_signals": int(get(config, "data.k_signals", 15)),
        "clip_samples": int(get(config, "model.clip_samples", 120)),
        "internal_filters": int(get(config, "model.internal_filters", 5)),
        "kernel_size": int(get(config, "model.kernel_size", 3)),
        "dropout": float(get(config, "model.dropout", 0.01)),
        "dilation_powers": get(config, "model.dilation_powers", None),
        "mlp_hidden": tuple(get(config, "model.mlp_hidden", [64])),
    }


def _teacher_from_checkpoint(checkpoint: dict, device: torch.device) -> TeacherPPGNetwork:
    model_config = dict(checkpoint.get("model_config", {}))
    model_config.pop("input_rows", None)
    teacher = TeacherPPGNetwork(**model_config).to(device)
    teacher.load_state_dict(checkpoint["model_state"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher


def train_epoch(
    teacher,
    student,
    loader,
    optimizer,
    device,
    lambda_f: float,
    lambda_df: float,
) -> dict[str, float]:
    teacher.eval()
    student.train()
    alignment_loss = NegativePearsonLoss()
    fidelity_loss = DataFidelityLoss()
    totals = {"total": 0.0, "feature": 0.0, "fidelity": 0.0}

    for batch in loader:
        rppg = batch["rppg"].to(device)
        ppg = batch["ppg"].to(device)
        target = batch["bp"].to(device)
        with torch.no_grad():
            teacher_features, _ = teacher(ppg)
        student_features, prediction = student(rppg)
        loss_f = alignment_loss(student_features, teacher_features)
        loss_df = fidelity_loss(prediction, target)
        loss = lambda_f * loss_f + lambda_df * loss_df

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = len(rppg)
        totals["total"] += float(loss.item()) * batch_size
        totals["feature"] += float(loss_f.item()) * batch_size
        totals["fidelity"] += float(loss_df.item()) * batch_size

    return {key: value / len(loader.dataset) for key, value in totals.items()}


@torch.no_grad()
def evaluate_video_level(student, loader, device) -> dict[str, float]:
    student.eval()
    records: list[dict] = []
    for batch in loader:
        _, prediction = student(batch["rppg"].to(device))
        prediction = prediction.cpu().numpy()
        target = batch["bp"].numpy()
        quality = batch["quality"].numpy()
        for index in range(len(prediction)):
            records.append(
                {
                    "video_key": batch["video_key"][index],
                    "dataset": batch["dataset"][index],
                    "subject_id": batch["subject_id"][index],
                    "video_id": batch["video_id"][index],
                    "prediction": prediction[index],
                    "target": target[index],
                    "quality": float(quality[index]),
                }
            )
    fused = quality_weighted_fusion(records)
    predictions = np.asarray([[row["SBP_pred"], row["DBP_pred"]] for row in fused])
    targets = np.asarray([[row["SBP_true"], row["DBP_true"]] for row in fused])
    return bp_metrics(predictions, targets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ALIVE student N_rPPG with a frozen PPG teacher.")
    parser.add_argument("--config", default="configs/student.yaml")
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--quality")
    parser.add_argument("--out-dir")
    parser.add_argument("--split-file")
    parser.add_argument("--resume")
    args = parser.parse_args()

    config = load_config(args.config)
    data_root = Path(args.data_root or get(config, "data.root", "processed/student"))
    quality = args.quality or str(get(config, "data.quality", "HQ"))
    out_dir = Path(args.out_dir or get(config, "output.dir", f"runs/student/{quality}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(get(config, "training.seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_path = Path(args.split_file) if args.split_file else out_dir / "subject_split.json"
    if split_path.exists():
        split = SplitDefinition.load(split_path)
    else:
        split = make_student_split(
            data_root,
            quality,
            clip_seconds=int(get(config, "data.clip_seconds", 4)),
            train_ratio=float(get(config, "data.train_ratio", 0.8)),
            val_ratio=float(get(config, "data.val_ratio", 0.1)),
            seed=seed,
        )
        split.save(split_path)

    dataset_kwargs = {
        "data_root": data_root,
        "quality": quality,
        "clip_seconds": int(get(config, "data.clip_seconds", 4)),
        "fps": int(get(config, "data.fps", 30)),
        "k_signals": int(get(config, "data.k_signals", 15)),
        "require_ppg": True,
    }
    train_set = StudentClipDataset(subjects=split.train, **dataset_kwargs)
    val_subjects = split.validation or split.test
    if not val_subjects:
        raise RuntimeError("Student training requires at least one validation or test subject")
    val_set = StudentClipDataset(subjects=val_subjects, **dataset_kwargs)

    batch_size = int(get(config, "training.batch_size", 4))
    num_workers = int(get(config, "training.num_workers", 0))
    train_loader = make_loader(train_set, batch_size, True, num_workers)
    val_loader = make_loader(val_set, batch_size, False, num_workers)

    teacher_checkpoint = load_checkpoint(args.teacher_checkpoint, device)
    teacher = _teacher_from_checkpoint(teacher_checkpoint, device)
    student = StudentRPPGNetwork(**_student_kwargs(config)).to(device)
    if student.clip_samples != teacher.clip_samples:
        raise ValueError(
            f"Teacher/student feature lengths differ: {teacher.clip_samples} vs {student.clip_samples}"
        )

    optimizer = Adam(student.parameters(), lr=float(get(config, "training.learning_rate", 1e-4)))
    start_epoch = 1
    best_score = float("inf")
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        student.load_state_dict(checkpoint["model_state"], strict=True)
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_score = float(checkpoint.get("best_score", best_score))

    lambda_f = float(get(config, "loss.lambda_f", 1.0))
    lambda_df = float(get(config, "loss.lambda_df", 1.0))
    epochs = int(get(config, "training.epochs", 20))
    history: list[dict] = []

    for epoch in range(start_epoch, epochs + 1):
        losses = train_epoch(
            teacher,
            student,
            train_loader,
            optimizer,
            device,
            lambda_f,
            lambda_df,
        )
        metrics = evaluate_video_level(student, val_loader, device)
        score = metrics["SBP_MAE"] + metrics["DBP_MAE"]
        row = {"epoch": epoch, **{f"loss_{k}": v for k, v in losses.items()}, **metrics}
        history.append(row)
        print(
            f"Epoch {epoch:03d} | L_C={losses['total']:.4f} | "
            f"L_F={losses['feature']:.4f} | L_DF={losses['fidelity']:.4f} | "
            f"video SBP MAE={metrics['SBP_MAE']:.3f} | video DBP MAE={metrics['DBP_MAE']:.3f}"
        )

        checkpoint = {
            "model_state": student.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": student.model_config(),
            "epoch": epoch,
            "best_score": min(best_score, score),
            "validation_metrics": metrics,
            "quality": quality,
            "source_config": config,
        }
        torch.save(checkpoint, out_dir / "last_student.pt")
        if score < best_score:
            best_score = score
            torch.save(checkpoint, out_dir / "best_student.pt")

    save_json({"history": history}, out_dir / "training_history.json")
    print(f"Best student checkpoint: {out_dir / 'best_student.pt'}")


if __name__ == "__main__":
    main()
