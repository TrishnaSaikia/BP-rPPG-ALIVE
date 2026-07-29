from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import get, load_config
from .data import SplitDefinition, StudentClipDataset, make_loader
from .metrics import bp_metrics, quality_weighted_fusion
from .models import StudentRPPGNetwork
from .utils import load_checkpoint, save_json, set_seed


def _student_from_checkpoint(checkpoint: dict, device: torch.device) -> StudentRPPGNetwork:
    model_config = dict(checkpoint.get("model_config", {}))
    input_rows = int(model_config.pop("input_rows"))
    student = StudentRPPGNetwork(k_signals=input_rows, **model_config).to(device)
    student.load_state_dict(checkpoint["model_state"], strict=True)
    student.eval()
    return student


@torch.no_grad()
def predict(student, loader, device) -> tuple[list[dict], list[dict], dict[str, float]]:
    clip_rows: list[dict] = []
    fusion_records: list[dict] = []
    for batch in loader:
        _, prediction = student(batch["rppg"].to(device))
        prediction = prediction.cpu().numpy()
        target = batch["bp"].numpy()
        quality = batch["quality"].numpy()
        for index in range(len(prediction)):
            row = {
                "dataset": batch["dataset"][index],
                "subject_id": batch["subject_id"][index],
                "video_id": batch["video_id"][index],
                "video_key": batch["video_key"][index],
                "clip_index": int(batch["clip_index"][index]),
                "clip_quality": float(quality[index]),
                "SBP_pred": float(prediction[index, 0]),
                "DBP_pred": float(prediction[index, 1]),
                "SBP_true": float(target[index, 0]),
                "DBP_true": float(target[index, 1]),
            }
            clip_rows.append(row)
            fusion_records.append(
                {
                    **row,
                    "prediction": prediction[index],
                    "target": target[index],
                    "quality": float(quality[index]),
                }
            )

    video_rows = quality_weighted_fusion(fusion_records)
    predictions = np.asarray([[row["SBP_pred"], row["DBP_pred"]] for row in video_rows])
    targets = np.asarray([[row["SBP_true"], row["DBP_true"]] for row in video_rows])
    metrics = bp_metrics(predictions, targets)
    return clip_rows, video_rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the trained ALIVE student. The PPG teacher is not loaded during testing."
    )
    parser.add_argument("--config", default="configs/student.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--quality")
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--split-file")
    parser.add_argument("--split-name", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--out-dir")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(get(config, "training.seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(args.checkpoint, device)
    student = _student_from_checkpoint(checkpoint, device)
    data_root = Path(args.data_root or get(config, "data.root", "processed/student"))
    quality = args.quality or str(checkpoint.get("quality", get(config, "data.quality", "HQ")))

    if args.subjects:
        subjects = args.subjects
    elif args.split_file:
        split = SplitDefinition.load(args.split_file)
        subjects = getattr(split, args.split_name)
    else:
        base = data_root / quality / f"{int(get(config, 'data.clip_seconds', 4))}_sec"
        subjects = sorted(path.name for path in base.iterdir() if path.is_dir())
    if not subjects:
        raise RuntimeError("No subjects selected for testing")

    dataset = StudentClipDataset(
        data_root=data_root,
        quality=quality,
        subjects=subjects,
        clip_seconds=int(get(config, "data.clip_seconds", 4)),
        fps=int(get(config, "data.fps", 30)),
        k_signals=student.input_rows,
        require_ppg=False,
    )
    loader = make_loader(
        dataset,
        batch_size=int(get(config, "testing.batch_size", get(config, "training.batch_size", 4))),
        shuffle=False,
        num_workers=int(get(config, "training.num_workers", 0)),
    )

    clip_rows, video_rows, metrics = predict(student, loader, device)
    out_dir = Path(args.out_dir or get(config, "testing.output_dir", f"results/{quality}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(clip_rows).to_csv(out_dir / "clip_predictions.csv", index=False)
    pd.DataFrame(video_rows).to_csv(out_dir / "video_predictions.csv", index=False)
    save_json(metrics, out_dir / "metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
