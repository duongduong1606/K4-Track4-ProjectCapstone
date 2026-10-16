# Vehicle exterior damage data audit

Audit date: 2026-10-16

## Final working set

| Dataset | Purpose | Images kept | Status |
|---|---|---:|---|
| AutoDamageIQ selected | Damage detection supplement | 5,633 | Valid YOLO labels, exact-deduplicated |
| Jijo clean | Dent/scratch detector | 2,159 | Re-split 80/10/10, no confirmed cross-split duplicate |
| CarDamagePanel | Vehicle panel segmentation | 560 | Valid segmentation labels; five leaked test copies removed |
| **Total** | Two-stage pipeline | **8,352** | Inside the 8k–10k target |

The staging downloads, unfinished Git LFS pointers, and redundant copies were removed after the selected dataset was verified. The retained files live under `data-audit/` and the audit reports live under `data-audit/reports/`.

## AutoDamageIQ findings

Source: <https://huggingface.co/datasets/tugberkkalay/autodamageiq-vehicle-damage-dataset>

The dataset card does not match the repository contents:

- Card claim: 13,089 train images, 2,094 validation images, about 42,419 labels.
- Repository reality: 10,000 train images, 2,094 validation images, 28,148 labels.
- Repository composition: 11,350 VehiDE-named images and 744 `existing_damage` images.
- Exact duplicate audit: 187 groups and 402 redundant files. Of those, 400 redundant files came from the 744-image `existing_damage` subset.
- One exact train/validation leak was found.
- One validation image had no matching label.
- The original validation split contains no `lamp_broken` or `tire_flat` instances, so it is not a valid six-class benchmark.

The selected local subset contains 5,633 readable, labelled, exact-unique images and 13,445 boxes:

| Class id | Meaning | Instances |
|---:|---|---:|
| 0 | crack | 792 |
| 1 | dent | 3,306 |
| 2 | glass shatter | 756 |
| 3 | lamp broken | 13 |
| 4 | scratch | 8,578 |
| 5 | tire flat | 0 |

Median image size is 640 × 480. All selected labels passed normalized YOLO-coordinate validation. A second perceptual check found three hash candidates but none passed the image-difference confirmation threshold, so no confirmed near-duplicate remains in this subset.

Manual inspection found useful real damage examples, but also inspection-marker writing, strong reflections, visible licence plates, and some questionable boxes. Use this data for training augmentation, not as the final product benchmark. Although the wrapper declares CC BY 4.0, its own card says the content comes from CarDD, VehiDE, and HITL; VehiDE's publication restricts use to non-commercial research/education. Do not assume the wrapper can relicense all underlying images for commercial use.

## Jijo findings and clean split

Raw data contained 2,427 images. The audit found:

- 33 exact duplicate groups.
- One exact train/test leak.
- 282 visually confirmed redundant near-duplicate pairs/groups after perceptual-hash candidate verification.
- 268 redundant images removed after clustering.

The dataset was rebuilt by duplicate group, before augmentation, with deterministic seed 42:

| Split | Images | Dent instances | Scratch instances |
|---|---:|---:|---:|
| Train | 1,727 | 1,530 | 3,410 |
| Validation | 216 | 169 | 470 |
| Test | 216 | 183 | 408 |

Post-clean verification: zero exact duplicates and zero confirmed near-duplicates across splits.

## CarDamagePanel findings

Source: <https://huggingface.co/datasets/kadireks/cardamagepanel-originals>

- 560 images: 415 train, 103 validation, 42 test after removing five confirmed test duplicates.
- 7,781 panel polygons across 21 classes.
- No exact duplicate.
- Five resized/cropped versions of the same source image were found across splits by perceptual hashing. Per the MVP cleanup policy, the five copies in `test` were removed together with their labels; the retained test set has 42 images.
- Post-clean audit: zero exact duplicates, zero confirmed near-duplicates across splits, and zero unreadable images (`reports/cardamagepanel/post_cleanup_duplicates.json`).

## Recommended MVP taxonomy

For a two-hour local model, train only these three output groups:

1. `dent`
2. `scratch`
3. `broken_or_cracked` — map crack and glass-shatter here; keep the 13 lamp examples out of evaluation because the class is too small.

Use CarDamagePanel as a separate panel-localization model. Do not merge panel labels into damage labels.

## Reproducibility scripts

- `scripts/audit_yolo_dataset.py`: image/label integrity, class counts, SHA-256 and perceptual duplicate audit.
- `scripts/audit_image_splits.py`: exact and confirmed near-duplicate checks across arbitrary splits.
- `scripts/rebuild_coco_splits.py`: duplicate-group-safe COCO re-splitting.
- `scripts/select_downloaded_yolo.py`: valid, labelled, exact-unique subset construction.
- `scripts/audit_yolo_labels.py`: label-only audit.
- `scripts/audit_lfs_pointers.py`: exact duplicate audit from Git LFS object hashes without downloading blobs.
