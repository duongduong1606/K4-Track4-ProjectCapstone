"""Build a three-class YOLO detection dataset without duplicating image bytes.

Sources:
* AutoDamageIQ selected (YOLO): crack/dent/glass/lamp/scratch -> 3 MVP classes.
* Jijo clean (COCO): dent/scratch -> the same class IDs.

On Windows, images are hard-linked when possible and copied only as a fallback.
The AutoDamageIQ split is deterministic; the already-clean Jijo split is kept.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUTO_CLASS_MAP = {0: 2, 1: 0, 2: 2, 3: 2, 4: 1}  # tire_flat (5) is excluded
JIJO_CLASS_MAP = {0: 0, 1: 1}


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_label(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def prepare_auto(root: Path, output: Path, seed: int) -> Counter:
    image_root = root / "data-audit" / "autodamageiq-selected" / "images" / "all"
    label_root = root / "data-audit" / "autodamageiq-selected" / "labels" / "all"
    items = sorted(p for p in image_root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    random.Random(seed).shuffle(items)
    n = len(items)
    train_end, val_end = round(n * 0.8), round(n * 0.9)
    stats: Counter = Counter()

    for index, image in enumerate(items):
        split = "train" if index < train_end else "val" if index < val_end else "test"
        source_label = label_root / f"{image.stem}.txt"
        rows: list[str] = []
        for raw in source_label.read_text(encoding="utf-8").splitlines():
            parts = raw.split()
            source_class = int(parts[0])
            if source_class not in AUTO_CLASS_MAP:
                continue
            target_class = AUTO_CLASS_MAP[source_class]
            rows.append(" ".join([str(target_class), *parts[1:]]))
            stats[f"instances_{target_class}"] += 1
        if not rows:
            stats["images_skipped_without_mvp_labels"] += 1
            continue
        name = f"auto_{image.name}"
        link_or_copy(image, output / "images" / split / name)
        write_label(output / "labels" / split / f"{Path(name).stem}.txt", rows)
        stats[f"images_{split}"] += 1
    return stats


def prepare_jijo(root: Path, output: Path) -> Counter:
    source_root = root / "data-audit" / "jijo-clean"
    stats: Counter = Counter()
    split_map = {"train": "train", "valid": "val", "test": "test"}

    for source_split, target_split in split_map.items():
        folder = source_root / source_split
        coco = json.loads((folder / "_annotations.coco.json").read_text(encoding="utf-8"))
        images = {item["id"]: item for item in coco["images"]}
        annotations: dict[int, list[dict]] = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations[annotation["image_id"]].append(annotation)

        for image_id, item in images.items():
            source_image = folder / item["file_name"]
            if not source_image.exists():
                stats["missing_images"] += 1
                continue
            width, height = float(item["width"]), float(item["height"])
            rows: list[str] = []
            for annotation in annotations.get(image_id, []):
                category = int(annotation["category_id"])
                if category not in JIJO_CLASS_MAP:
                    continue
                x, y, w, h = map(float, annotation["bbox"])
                xc = (x + w / 2) / width
                yc = (y + h / 2) / height
                rows.append(
                    f"{JIJO_CLASS_MAP[category]} {xc:.6f} {yc:.6f} "
                    f"{w / width:.6f} {h / height:.6f}"
                )
                stats[f"instances_{JIJO_CLASS_MAP[category]}"] += 1
            if not rows:
                stats["images_skipped_without_mvp_labels"] += 1
                continue
            name = f"jijo_{source_image.name}"
            link_or_copy(source_image, output / "images" / target_split / name)
            write_label(output / "labels" / target_split / f"{Path(name).stem}.txt", rows)
            stats[f"images_{target_split}"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "data" / "processed" / "damage").resolve()

    if output.exists() and args.force:
        shutil.rmtree(output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output is not empty: {output}. Use --force to rebuild it.")

    output.mkdir(parents=True, exist_ok=True)
    stats = prepare_auto(root, output, args.seed) + prepare_jijo(root, output)
    report = {
        "seed": args.seed,
        "output": str(output),
        "class_names": {"0": "dent", "1": "scratch", "2": "broken_or_cracked"},
        "stats": dict(sorted(stats.items())),
    }
    (output / "build_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
