"""Read-only raw-data audit; reports are written outside the dataset tree.

Use --limit for a bounded sample. --all explicitly opts into a full audit.
Masks and alternative representations are counted as image files, not unique samples.
"""
import argparse
import csv
import itertools
import json
import os
from collections import Counter
from pathlib import Path
from PIL import Image
from src.utils.config import configured_path, ensure_output_path, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "Datasets"


def get_image_files(root):
    extensions = set(load_config("preprocessing")["images"]["extensions"])
    for directory, subdirs, filenames in os.walk(root):
        subdirs.sort()
        for name in sorted(filenames):
            path = Path(directory) / name
            if path.suffix.lower() in extensions:
                yield path


def detect_split(path):
    parts = {part.lower() for part in Path(path).parts}
    for names, label in [({"train", "training"}, "train"), ({"val", "valid", "validation"}, "validation"), ({"test", "testing"}, "test")]:
        if parts & names:
            return label
    return "unknown"


def audit_image(path):
    try:
        with Image.open(path) as image:
            width, height, mode, fmt = *image.size, image.mode, image.format
            image.verify()
        return dict(status="ok", width=width, height=height, mode=mode, format=fmt, error="")
    except Exception as error:
        return dict(status="unreadable", width="", height="", mode="", format="", error=str(error))


def audit(root=None, output=None, limit=None):
    root = Path(root) if root is not None else DATASET_ROOT
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}. See README Dataset setup.")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    output = ensure_output_path(output if output is not None else configured_path("processed_root") / "audit")
    if output == root or root in output.parents:
        raise ValueError("Audit reports cannot be written inside raw data")
    targets = [output / name for name in ("dataset_inventory.csv", "dataset_summary.json", "audit_report.txt")]
    if any(p.exists() for p in targets):
        raise FileExistsError("Audit reports already exist; choose a fresh --output directory")
    output.mkdir(parents=True, exist_ok=True)
    counters = {name: Counter() for name in ("datasets", "splits", "extensions", "image_status", "resolutions")}
    fields = ["image_path", "dataset", "split", "folder", "role", "extension", "width", "height", "mode", "format", "status", "error"]
    images = get_image_files(root)
    if limit is not None:
        images = itertools.islice(images, limit)
    total = 0
    with targets[0].open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for path in images:
            rel = path.relative_to(root)
            dataset = rel.parts[0]
            split = detect_split(rel)
            role = "mask" if dataset == "plantseg" and "annotations" in rel.parts else "image_or_representation"
            info = audit_image(path)
            writer.writerow(dict(image_path=(Path("Datasets") / rel).as_posix(), dataset=dataset,
                                 split=split, folder=rel.parent.as_posix(), role=role,
                                 extension=path.suffix.lower(), **info))
            total += 1
            for name, value in (("datasets",dataset),("splits",split),("extensions",path.suffix.lower()),("image_status",info["status"])):
                counters[name][value] += 1
            if info["width"]:
                counters["resolutions"][f"{info['width']}x{info['height']}"] += 1
    summary = dict(scope="full" if limit is None else "bounded_sample", limit=limit, total_images=total,
                   counting_unit="image files, including masks and alternate representations",
                   **{key: dict(value) for key, value in counters.items()})
    targets[1].write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    targets[2].write_text("FOLTRA DATASET AUDIT\n"+json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--limit", type=int)
    scope.add_argument("--all", action="store_true")
    parser.add_argument("--output", help="Fresh repository-relative report directory")
    args = parser.parse_args()
    result = audit(root=configured_path("dataset_root"), output=args.output, limit=args.limit)
    print(json.dumps({key: result[key] for key in ("scope", "total_images", "image_status")}, indent=2))


if __name__ == "__main__":
    main()
