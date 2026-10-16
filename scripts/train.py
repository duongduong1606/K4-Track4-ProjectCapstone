"""Train either model in the balanced exterior-damage stack."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("damage", "panel"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true", help="Resume runs/<name>/weights/last.pt")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    if args.task == "damage":
        model_name, data_name = "yolo26n.pt", "damage.yaml"
        epochs = args.epochs or 60
        batch = args.batch or 16
        patience = args.patience or 12
    else:
        model_name, data_name = "yolo26n-seg.pt", "panel.yaml"
        epochs = args.epochs or 100
        batch = args.batch or 8
        patience = args.patience or 20

    run_name = args.name or args.task
    if args.resume:
        checkpoint = root / "runs" / run_name / "weights" / "last.pt"
        if not checkpoint.exists():
            raise SystemExit(f"Resume checkpoint does not exist: {checkpoint}")
        YOLO(checkpoint).train(resume=True)
        return

    model = YOLO(root / "models" / model_name)
    model.train(
        data=root / "configs" / data_name,
        epochs=epochs,
        imgsz=args.imgsz,
        batch=batch,
        workers=args.workers,
        device=args.device,
        project=root / "runs",
        name=run_name,
        exist_ok=True,
        pretrained=True,
        patience=patience,
        amp=True,
        cache=False,
        fraction=args.fraction,
        seed=42,
        deterministic=True,
        val=args.val,
        plots=args.plots,
    )


if __name__ == "__main__":
    main()
