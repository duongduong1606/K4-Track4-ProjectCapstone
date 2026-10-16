#!/usr/bin/env python3
"""Audit a YOLO detection dataset and write reproducible JSON/CSV reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def dhash(image: Image.Image, size: int = 8) -> int:
    gray = ImageOps.grayscale(image).resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = list(gray.get_flattened_data())
    value = 0
    for y in range(size):
        row = y * (size + 1)
        for x in range(size):
            value = (value << 1) | (pixels[row + x] > pixels[row + x + 1])
    return value


def split_for(path: Path, images_root: Path) -> str:
    rel = path.relative_to(images_root)
    return rel.parts[0] if len(rel.parts) > 1 else "unsplit"


def label_for(image_path: Path, images_root: Path, labels_root: Path) -> Path:
    return (labels_root / image_path.relative_to(images_root)).with_suffix(".txt")


def parse_label(path: Path) -> tuple[Counter, int, list[str]]:
    counts: Counter = Counter()
    boxes = 0
    errors: list[str] = []
    if not path.exists():
        return counts, boxes, ["missing_label"]
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        is_box = len(fields) == 5
        is_polygon = len(fields) >= 7 and len(fields) % 2 == 1
        if not (is_box or is_polygon):
            errors.append("invalid_field_count")
            continue
        try:
            class_id = int(fields[0])
            coords = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append("non_numeric")
            continue
        if class_id < 0:
            errors.append("negative_class")
        if any(value < 0 or value > 1 for value in coords):
            errors.append("coordinate_out_of_range")
        if is_box and (coords[2] <= 0 or coords[3] <= 0):
            errors.append("non_positive_box")
        counts[class_id] += 1
        boxes += 1
    return counts, boxes, errors


def find_near_pairs(rows: list[dict], max_distance: int) -> list[dict]:
    # Four 16-bit bands make candidate search fast while guaranteeing that
    # hashes within Hamming distance <= 3 share at least one band.
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    pairs: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for index, row in enumerate(rows):
        value = int(row["dhash"], 16)
        candidates: set[int] = set()
        for band in range(4):
            candidates.update(buckets[(band, (value >> (band * 16)) & 0xFFFF)])
        for other_index in candidates:
            key = (other_index, index)
            if key in seen:
                continue
            seen.add(key)
            other = rows[other_index]
            if row["sha256"] == other["sha256"]:
                continue
            distance = (value ^ int(other["dhash"], 16)).bit_count()
            if distance <= max_distance:
                pairs.append(
                    {
                        "a": other["path"],
                        "a_split": other["split"],
                        "b": row["path"],
                        "b_split": row["split"],
                        "hamming_distance": distance,
                        "cross_split": row["split"] != other["split"],
                    }
                )
        for band in range(4):
            buckets[(band, (value >> (band * 16)) & 0xFFFF)].append(index)
    return pairs


def audit(root: Path, output_dir: Path, near_distance: int) -> dict:
    images_root = root / "images"
    labels_root = root / "labels"
    image_paths = sorted(
        path for path in images_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    rows: list[dict] = []
    split_images: Counter = Counter()
    split_boxes: Counter = Counter()
    class_counts: Counter = Counter()
    error_counts: Counter = Counter()
    dimensions: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for image_path in image_paths:
        split = split_for(image_path, images_root)
        rel_path = image_path.relative_to(root).as_posix()
        content = image_path.read_bytes()
        image_errors: list[str] = []
        width = height = 0
        hash_value = 0
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                hash_value = dhash(image)
        except Exception as exc:  # Pillow raises several format-specific exceptions.
            image_errors.append(f"unreadable_image:{type(exc).__name__}")
        label_counts, boxes, label_errors = parse_label(label_for(image_path, images_root, labels_root))
        errors = image_errors + label_errors
        split_images[split] += 1
        split_boxes[split] += boxes
        class_counts.update(label_counts)
        error_counts.update(errors)
        if width and height:
            dimensions[split].append((width, height))
        rows.append(
            {
                "path": rel_path,
                "split": split,
                "width": width,
                "height": height,
                "boxes": boxes,
                "classes": json.dumps(dict(sorted(label_counts.items()))),
                "sha256": hashlib.sha256(content).hexdigest(),
                "dhash": f"{hash_value:016x}",
                "errors": ";".join(errors),
            }
        )

    exact_groups = []
    by_sha: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_sha[row["sha256"]].append(row)
    for sha256, group in by_sha.items():
        if len(group) > 1:
            splits = sorted({item["split"] for item in group})
            exact_groups.append(
                {
                    "sha256": sha256,
                    "paths": [item["path"] for item in group],
                    "splits": splits,
                    "cross_split": len(splits) > 1,
                }
            )

    near_pairs = find_near_pairs(rows, near_distance)
    dimension_summary = {}
    for split, values in dimensions.items():
        areas = [width * height for width, height in values]
        widths = [width for width, _ in values]
        heights = [height for _, height in values]
        dimension_summary[split] = {
            "min": [min(widths), min(heights)],
            "median": [int(statistics.median(widths)), int(statistics.median(heights))],
            "max": [max(widths), max(heights)],
            "median_megapixels": round(statistics.median(areas) / 1_000_000, 3),
        }

    report = {
        "dataset_root": str(root.resolve()),
        "image_count": len(rows),
        "images_by_split": dict(sorted(split_images.items())),
        "boxes_by_split": dict(sorted(split_boxes.items())),
        "boxes_by_class_id": {str(key): value for key, value in sorted(class_counts.items())},
        "dimension_summary": dimension_summary,
        "error_counts": dict(sorted(error_counts.items())),
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_cross_split_groups": sum(item["cross_split"] for item in exact_groups),
        "near_duplicate_pairs": len(near_pairs),
        "near_duplicate_cross_split_pairs": sum(item["cross_split"] for item in near_pairs),
        "near_duplicate_max_hamming_distance": near_distance,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["path"])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "exact_duplicates.json").write_text(json.dumps(exact_groups, indent=2), encoding="utf-8")
    (output_dir / "near_duplicates.json").write_text(json.dumps(near_pairs, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Dataset root containing images/ and labels/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--near-distance", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset, args.output, args.near_distance), indent=2))


if __name__ == "__main__":
    main()
