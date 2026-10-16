#!/usr/bin/env python3
"""Audit YOLO label files without requiring image files to be present."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    class_counts: Counter = Counter()
    split_class_counts: Counter = Counter()
    errors: Counter = Counter()
    files_by_split: Counter = Counter()
    empty_files = 0
    instances = 0
    for path in args.labels.rglob("*.txt"):
        split = path.relative_to(args.labels).parts[0]
        files_by_split[split] += 1
        nonempty = False
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            fields = raw.split()
            if not fields:
                continue
            nonempty = True
            if len(fields) != 5:
                errors["invalid_field_count"] += 1
                continue
            try:
                class_id = int(fields[0])
                coords = [float(value) for value in fields[1:]]
            except ValueError:
                errors["non_numeric"] += 1
                continue
            if class_id < 0:
                errors["negative_class"] += 1
            if any(value < 0 or value > 1 for value in coords):
                errors["coordinate_out_of_range"] += 1
            if coords[2] <= 0 or coords[3] <= 0:
                errors["non_positive_box"] += 1
            class_counts[class_id] += 1
            split_class_counts[(split, class_id)] += 1
            instances += 1
        if not nonempty:
            empty_files += 1

    report = {
        "label_files": sum(files_by_split.values()),
        "files_by_split": dict(sorted(files_by_split.items())),
        "empty_label_files": empty_files,
        "instances": instances,
        "class_counts": {str(key): value for key, value in sorted(class_counts.items())},
        "class_counts_by_split": {
            f"{split}:{class_id}": value
            for (split, class_id), value in sorted(split_class_counts.items())
        },
        "errors": dict(sorted(errors.items())),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
