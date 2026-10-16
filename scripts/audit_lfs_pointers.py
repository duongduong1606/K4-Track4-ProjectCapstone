#!/usr/bin/env python3
"""Count exact duplicates from Git LFS pointer files without downloading blobs."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-repo", type=Path, help="Read pointer blobs from Git HEAD instead of the checkout")
    args = parser.parse_args()

    by_oid: dict[str, list[dict]] = defaultdict(list)
    non_pointers = 0
    entries = []
    if args.git_repo:
        listing = subprocess.check_output(
            ["git", "ls-tree", "-r", "HEAD", "images"], cwd=args.git_repo, text=True
        )
        objects = []
        for line in listing.splitlines():
            metadata, path_text = line.split("\t", 1)
            objects.append((metadata.split()[2], path_text))
        batch = subprocess.check_output(
            ["git", "cat-file", "--batch"],
            cwd=args.git_repo,
            input="".join(f"{oid}\n" for oid, _ in objects).encode(),
        )
        offset = 0
        for _, path_text in objects:
            line_end = batch.index(b"\n", offset)
            header = batch[offset:line_end].decode().split()
            size = int(header[2])
            content_start = line_end + 1
            content = batch[content_start : content_start + size]
            offset = content_start + size + 1
            entries.append((path_text.removeprefix("images/"), content.decode("ascii", errors="ignore")))
    else:
        for path in args.images.rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="ascii", errors="ignore") if path.stat().st_size < 1024 else ""
                entries.append((path.relative_to(args.images).as_posix(), text))

    for rel, text in entries:
        if not text.startswith("version https://git-lfs.github.com/spec/v1"):
            non_pointers += 1
            continue
        fields = dict(line.split(" ", 1) for line in text.splitlines()[1:] if " " in line)
        oid = fields["oid"].removeprefix("sha256:")
        by_oid[oid].append(
            {
                "path": rel,
                "split": rel.split("/", 1)[0],
                "size": int(fields["size"]),
            }
        )

    groups = []
    for oid, items in by_oid.items():
        if len(items) < 2:
            continue
        splits = sorted({item["split"] for item in items})
        groups.append(
            {
                "sha256": oid,
                "copies": len(items),
                "splits": splits,
                "cross_split": len(splits) > 1,
                "items": items,
            }
        )
    groups.sort(key=lambda item: (-item["copies"], item["sha256"]))
    report = {
        "pointer_files": sum(len(items) for items in by_oid.values()),
        "downloaded_non_pointer_files": non_pointers,
        "unique_content_hashes": len(by_oid),
        "duplicate_groups": len(groups),
        "redundant_files": sum(item["copies"] - 1 for item in groups),
        "cross_split_groups": sum(item["cross_split"] for item in groups),
        "groups": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "groups"}, indent=2))


if __name__ == "__main__":
    main()
