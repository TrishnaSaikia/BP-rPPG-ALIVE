from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam

from .config import get, load_config
from .data import (
    SplitDefinition,
    TeacherPPGDataset,
    make_loader,
    make_teacher_split,
)
from .losses import DataFidelityLoss
from .metrics import bp_metrics
from .models import TeacherPPGNetwork
from .utils import load_checkpoint, save_json, set_seed


def _model_kwargs(config: dict) -> dict:
    return {
        "clip_samples": int(get(config, "model.clip_samples", 120)),
        "internal_filters": int(get(config, "model.internal_filters", 5)),
        "kernel_size": int(get(config, "model.kernel_size", 3)),
        "dropout": float(get(config, "model.dropout", 0.01)),
        "dilation_powers": get(config, "model.dilation_powers", None),
        "mlp_hidden": tuple(get(config, "model.mlp_hidden", [64])),
    }


def train_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        ppg = batch["ppg"].to(device)
        target = batch["bp"].to(device)
        _, prediction = model(ppg)
        loss = loss_fn(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * len(ppg)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in loader:
        _, prediction = model(batch["ppg"].to(device))
        predictions.append(prediction.cpu().numpy())
        targets.append(batch["bp"].numpy())
    return bp_metrics(np.concatenate(predictions), np.concatenate(targets))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PPG teacher network N_PPG.")
    parser.add_argument("--config", default="configs/teacher.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--out-dir")
    parser.add_argument("--split-file")
    parser.add_argument("--resume")
    args = parser.parse_args()

    config = load_config(args.config)
    data_root = Path(args.data_root or get(config, "data.root", "processed/teacher_ppg/4_sec"))
    out_dir = Path(args.out_dir or get(config, "output.dir", "runs/teacher"))
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(get(config, "training.seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_path = Path(args.split_file) if args.split_file else out_dir / "subject_split.json"
    if split_path.exists():
        split = SplitDefinition.load(split_path)
    else:
        split = make_teacher_split(
            data_root,
            train_ratio=float(get(config, "data.train_ratio", 0.8)),
            val_ratio=float(get(config, "data.val_ratio", 0.1)),
            seed=seed,
        )
        split.save(split_path)

    clip_samples = int(get(config, "model.clip_samples", 120))
    train_set = TeacherPPGDataset(data_root, split.train, clip_samples=clip_samples)
    val_paths = split.validation or split.test
    if not val_paths:
        raise RuntimeError("Teacher training requires at least one validation or test subject")
    val_set = TeacherPPGDataset(data_root, val_paths, clip_samples=clip_samples)

    batch_size = int(get(config, "training.batch_size", 4))
    num_workers = int(get(config, "training.num_workers", 0))
    train_loader = make_loader(train_set, batch_size, True, num_workers)
    val_loader = make_loader(val_set, batch_size, False, num_workers)

    model = TeacherPPGNetwork(**_model_kwargs(config)).to(device)
    optimizer = Adam(model.parameters(), lr=float(get(config, "training.learning_rate", 1e-4)))
    loss_fn = DataFidelityLoss()
    start_epoch = 1
    best_score = float("inf")

    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_score = float(checkpoint.get("best_score", best_score))

    epochs = int(get(config, "training.epochs", 20))
    history: list[dict] = []
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        metrics = evaluate(model, val_loader, device)
        score = metrics["SBP_MAE"] + metrics["DBP_MAE"]
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(row)
        print(
            f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
            f"SBP MAE={metrics['SBP_MAE']:.3f} | DBP MAE={metrics['DBP_MAE']:.3f}"
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": model.model_config(),
            "epoch": epoch,
            "best_score": min(best_score, score),
            "validation_metrics": metrics,
            "source_config": config,
        }
        torch.save(checkpoint, out_dir / "last_teacher.pt")
        if score < best_score:
            best_score = score
            torch.save(checkpoint, out_dir / "best_teacher.pt")

    save_json({"history": history}, out_dir / "training_history.json")
    print(f"Best teacher checkpoint: {out_dir / 'best_teacher.pt'}")


if __name__ == "__main__":
    main()
