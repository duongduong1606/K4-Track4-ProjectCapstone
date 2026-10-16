#!/usr/bin/env python3
"""Deduplicate COCO splits, then rebuild deterministic group-safe splits."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def allocate(records: list[dict], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    by_signature: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    for record in records:
        by_signature[tuple(sorted(record["classes"]))].append(record)
    result = {"train": [], "valid": [], "test": []}
    for signature in sorted(by_signature):
        group = by_signature[signature]
        rng.shuffle(group)
        count = len(group)
        train_end = round(count * train_ratio)
        val_end = train_end + round(count * val_ratio)
        result["train"].extend(group[:train_end])
        result["valid"].extend(group[train_end:val_end])
        result["test"].extend(group[val_end:])
    for split in result:
        rng.shuffle(result[split])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("duplicate_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train-ratio + val-ratio must be less than 1")

    records = []
    categories = None
    for split in args.splits:
        coco_path = args.source / split / "_annotations.coco.json"
        coco = json.loads(coco_path.read_text(encoding="utf-8-sig"))
        categories = categories or coco["categories"]
        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations_by_image[annotation["image_id"]].append(annotation)
        for image in coco["images"]:
            rel_path = f"{split}/{image['file_name']}"
            annotations = annotations_by_image[image["id"]]
            records.append(
                {
                    "path": rel_path,
                    "source_path": args.source / rel_path,
                    "image": image,
                    "annotations": annotations,
                    "classes": {annotation["category_id"] for annotation in annotations},
                }
            )

    by_path = {record["path"]: record for record in records}
    union_find = UnionFind(list(by_path))
    duplicate_data = json.loads(args.duplicate_report.read_text(encoding="utf-8"))
    for group in duplicate_data["exact_duplicate_groups"]:
        paths = [path for path in group["paths"] if path in by_path]
        for path in paths[1:]:
            union_find.union(paths[0], path)
    for pair in duplicate_data["near_duplicate_pairs"]:
        if pair.get("confirmed") and pair["a"] in by_path and pair["b"] in by_path:
            union_find.union(pair["a"], pair["b"])

    clusters: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        clusters[union_find.find(record["path"])].append(record)
    unique_records = []
    dropped = []
    for cluster in clusters.values():
        # Prefer the most completely annotated copy, then a stable path.
        cluster.sort(key=lambda item: (-len(item["annotations"]), item["path"]))
        unique_records.append(cluster[0])
        dropped.extend(item["path"] for item in cluster[1:])

    allocated = allocate(unique_records, args.train_ratio, args.val_ratio, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, split_records in allocated.items():
        split_dir = args.output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        images = []
        annotations = []
        class_counts: Counter = Counter()
        for image_id, record in enumerate(split_records, 1):
            destination = split_dir / record["source_path"].name
            if not destination.exists():
                try:
                    os.link(record["source_path"], destination)
                except OSError:
                    shutil.copy2(record["source_path"], destination)
            image = dict(record["image"])
            image["id"] = image_id
            image["file_name"] = destination.name
            images.append(image)
            for annotation in record["annotations"]:
                copied = dict(annotation)
                copied["id"] = len(annotations) + 1
                copied["image_id"] = image_id
                annotations.append(copied)
                class_counts[copied["category_id"]] += 1
        coco = {"images": images, "annotations": annotations, "categories": categories}
        (split_dir / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
        stats[split] = {
            "images": len(images),
            "annotations": len(annotations),
            "class_instances": {str(key): value for key, value in sorted(class_counts.items())},
        }

    report = {
        "source_images": len(records),
        "unique_images": len(unique_records),
        "dropped_duplicate_images": len(dropped),
        "ratios": {
            "train": args.train_ratio,
            "valid": args.val_ratio,
            "test": 1 - args.train_ratio - args.val_ratio,
        },
        "stats": stats,
        "dropped_paths": sorted(dropped),
    }
    (args.output / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "dropped_paths"}, indent=2))


if __name__ == "__main__":
    main()
