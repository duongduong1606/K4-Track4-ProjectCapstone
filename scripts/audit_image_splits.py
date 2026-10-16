#!/usr/bin/env python3
"""Find exact and perceptual duplicates across arbitrary image split folders."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageChops, ImageOps, ImageStat

from audit_yolo_dataset import IMAGE_SUFFIXES, dhash, find_near_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--near-distance", type=int, default=3)
    parser.add_argument("--max-mean-abs-diff", type=float, default=12.0)
    args = parser.parse_args()

    rows = []
    unreadable = []
    for split in args.splits:
        split_root = args.root / split
        for path in sorted(split_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                content = path.read_bytes()
                with Image.open(path) as image:
                    image.load()
                    hash_value = dhash(image)
                rows.append(
                    {
                        "path": path.relative_to(args.root).as_posix(),
                        "split": split,
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "dhash": f"{hash_value:016x}",
                    }
                )
            except Exception as exc:
                unreadable.append({"path": str(path), "error": type(exc).__name__})

    by_sha = defaultdict(list)
    for row in rows:
        by_sha[row["sha256"]].append(row)
    exact_groups = []
    for digest, group in by_sha.items():
        if len(group) < 2:
            continue
        splits = sorted({item["split"] for item in group})
        exact_groups.append(
            {
                "sha256": digest,
                "paths": [item["path"] for item in group],
                "splits": splits,
                "cross_split": len(splits) > 1,
            }
        )

    near_pairs = find_near_pairs(rows, args.near_distance)
    for pair in near_pairs:
        with Image.open(args.root / pair["a"]) as left, Image.open(args.root / pair["b"]) as right:
            left_gray = ImageOps.grayscale(left).resize((128, 128), Image.Resampling.LANCZOS)
            right_gray = ImageOps.grayscale(right).resize((128, 128), Image.Resampling.LANCZOS)
            mean_abs_diff = ImageStat.Stat(ImageChops.difference(left_gray, right_gray)).mean[0]
        pair["mean_abs_diff"] = round(mean_abs_diff, 3)
        pair["confirmed"] = mean_abs_diff <= args.max_mean_abs_diff
    report = {
        "image_count": len(rows),
        "unreadable": unreadable,
        "exact_duplicate_groups": exact_groups,
        "exact_cross_split_groups": sum(item["cross_split"] for item in exact_groups),
        "near_duplicate_pairs": near_pairs,
        "near_cross_split_pairs": sum(item["cross_split"] for item in near_pairs),
        "confirmed_near_pairs": sum(item["confirmed"] for item in near_pairs),
        "confirmed_near_cross_split_pairs": sum(
            item["confirmed"] and item["cross_split"] for item in near_pairs
        ),
        "near_distance": args.near_distance,
        "max_mean_abs_diff": args.max_mean_abs_diff,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "image_count": len(rows),
                "exact_duplicate_groups": len(exact_groups),
                "exact_cross_split_groups": report["exact_cross_split_groups"],
                "near_duplicate_pairs": len(near_pairs),
                "near_cross_split_pairs": report["near_cross_split_pairs"],
                "confirmed_near_pairs": report["confirmed_near_pairs"],
                "confirmed_near_cross_split_pairs": report["confirmed_near_cross_split_pairs"],
                "unreadable": len(unreadable),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
