"""Generate reproducible local evidence for trained models and demo test cases."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from demo_app.main import DAMAGE_MODEL_PATH, PANEL_MODEL_PATH, ROOT, diagnose  # noqa: E402


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def best_training_row(path: Path, metric: str) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No training rows in {path}")
    row = max(rows, key=lambda item: float(item[metric]))
    return {
        "epoch": int(row["epoch"]),
        "elapsed_seconds": round(float(rows[-1]["time"]), 3),
        "metric": metric,
        "metric_value": round(float(row[metric]), 6),
        "row": {key: round(float(value), 6) for key, value in row.items() if key != "epoch"},
    }


def evaluate_models(output: Path) -> dict[str, Any]:
    validation_root = output / "ultralytics"
    damage_metrics = YOLO(DAMAGE_MODEL_PATH).val(
        data=ROOT / "configs" / "damage.yaml",
        split="test",
        imgsz=640,
        batch=16,
        workers=4,
        device=0,
        plots=False,
        verbose=False,
        project=validation_root,
        name="damage-test",
        exist_ok=True,
    )
    panel_metrics = YOLO(PANEL_MODEL_PATH).val(
        data=ROOT / "configs" / "panel.yaml",
        split="test",
        imgsz=640,
        batch=8,
        workers=4,
        device=0,
        plots=False,
        verbose=False,
        project=validation_root,
        name="panel-test",
        exist_ok=True,
    )
    return {
        "damage_test": native(damage_metrics.results_dict),
        "panel_test": native(panel_metrics.results_dict),
    }


def evaluate_cases(test_cases: Path, output: Path, confidence: float) -> list[dict[str, Any]]:
    annotated_root = output / "annotated"
    annotated_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    image_paths = sorted(
        path for path in test_cases.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    for image_path in image_paths:
        raw = image_path.read_bytes()
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            rows.append({"case": image_path.stem, "file": image_path.name, "error": "unreadable"})
            continue
        result = diagnose(image, confidence)
        encoded = result.pop("annotated_image").split(",", 1)[1]
        annotated_path = annotated_root / f"{image_path.stem}.jpg"
        annotated_path.write_bytes(base64.b64decode(encoded))
        rows.append(
            {
                "case": image_path.stem,
                "file": image_path.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "image_size": result["image_size"],
                "finding_count": result["count"],
                "findings": result["diagnoses"],
                "annotated_local": str(annotated_path.relative_to(ROOT)).replace("\\", "/"),
            }
        )
    return rows


def write_manifest(path: Path, cases: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case", "file", "sha256", "width", "height", "finding_count", "findings"),
        )
        writer.writeheader()
        for item in cases:
            size = item.get("image_size", {})
            findings = "; ".join(
                f"{finding['damage']}@{finding['panel'] or 'unknown'}:{finding['confidence']:.3f}"
                for finding in item.get("findings", [])
            )
            writer.writerow(
                {
                    "case": item["case"],
                    "file": item["file"],
                    "sha256": item.get("sha256", ""),
                    "width": size.get("width", ""),
                    "height": size.get("height", ""),
                    "finding_count": item.get("finding_count", ""),
                    "findings": findings,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--skip-dataset-eval", action="store_true")
    args = parser.parse_args()

    output = ROOT / "reports" / "evidence"
    output.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, Any] = {
        "configuration": {
            "confidence": args.confidence,
            "damage_model": str(DAMAGE_MODEL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "panel_model": str(PANEL_MODEL_PATH.relative_to(ROOT)).replace("\\", "/"),
            "image_size": 640,
        },
        "training": {
            "damage": best_training_row(
                ROOT / "runs" / "damage" / "results.csv", "metrics/mAP50-95(B)"
            ),
            "panel": best_training_row(
                ROOT / "runs" / "panel" / "results.csv", "metrics/mAP50-95(M)"
            ),
        },
        "test_cases": evaluate_cases(ROOT / "test_cases", output, args.confidence),
    }
    if not args.skip_dataset_eval:
        evidence["evaluation"] = evaluate_models(output)

    (output / "results.json").write_text(
        json.dumps(native(evidence), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_manifest(output / "manifest.csv", evidence["test_cases"])
    print(json.dumps({
        "cases": len(evidence["test_cases"]),
        "findings": sum(item.get("finding_count", 0) for item in evidence["test_cases"]),
        "results": str(output / "results.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
