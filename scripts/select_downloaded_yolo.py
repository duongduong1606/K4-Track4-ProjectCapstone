#!/usr/bin/env python3
"""Build a hard-linked, valid and exact-deduplicated YOLO subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from audit_yolo_dataset import IMAGE_SUFFIXES, parse_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=10_000)
    args = parser.parse_args()

    candidates = []
    excluded = []
    seen_hashes = {}
    for image_path in sorted((args.source / "images").rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        split = image_path.relative_to(args.source / "images").parts[0]
        label_path = args.source / "labels" / split / f"{image_path.stem}.txt"
        if image_path.stat().st_size < 200:
            excluded.append({"path": str(image_path), "reason": "lfs_pointer_not_downloaded"})
            continue
        label_counts, boxes, errors = parse_label(label_path)
        if errors:
            excluded.append({"path": str(image_path), "reason": ",".join(sorted(set(errors)))})
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            excluded.append({"path": str(image_path), "reason": f"unreadable:{type(exc).__name__}"})
            continue
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest in seen_hashes:
            excluded.append(
                {"path": str(image_path), "reason": "exact_duplicate", "duplicate_of": seen_hashes[digest]}
            )
            continue
        seen_hashes[digest] = str(image_path)
        candidates.append(
            {
                "image": image_path,
                "label": label_path,
                "split": split,
                "boxes": boxes,
                "classes": label_counts,
                "sha256": digest,
            }
        )

    # Rare-class examples sort first if a cap is needed; the final ordering is
    # deterministic and does not silently discard lamp/tire examples.
    global_counts = Counter()
    for item in candidates:
        global_counts.update(item["classes"])
    candidates.sort(
        key=lambda item: (
            min((global_counts[class_id] for class_id in item["classes"]), default=10**9),
            str(item["image"]),
        )
    )
    selected = candidates[: args.limit]

    images_out = args.output / "images" / "all"
    labels_out = args.output / "labels" / "all"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    selected_class_counts: Counter = Counter()
    manifest = []
    for index, item in enumerate(selected, 1):
        name = f"{index:05d}_{item['split']}_{item['image'].name}"
        image_out = images_out / name
        label_out = labels_out / f"{Path(name).stem}.txt"
        for source, destination in ((item["image"], image_out), (item["label"], label_out)):
            if not destination.exists():
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
        selected_class_counts.update(item["classes"])
        manifest.append(
            {
                "image": image_out.relative_to(args.output).as_posix(),
                "source": str(item["image"]),
                "source_split": item["split"],
                "boxes": item["boxes"],
                "sha256": item["sha256"],
            }
        )

    report = {
        "valid_unique_candidates": len(candidates),
        "selected_images": len(selected),
        "selected_instances": sum(item["boxes"] for item in selected),
        "selected_class_counts": {
            str(key): value for key, value in sorted(selected_class_counts.items())
        },
        "excluded_count": len(excluded),
        "excluded_reasons": dict(Counter(item["reason"] for item in excluded)),
        "manifest": manifest,
        "excluded": excluded,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "selection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"manifest", "excluded"}}, indent=2))


if __name__ == "__main__":
    main()
