"""Migrated diagnostic; TOBRFV semantics are provisional. See docs/datasets/tobrfv.md."""
from src.utils.config import dataset_path, manifest_path, resolve_image_path
import csv
import os
import sys
from collections import Counter


def load_split(path):
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            samples.append(row)

    return samples


def verify_split(dataset_name):
    base = str(manifest_path("."))

    files = {
        "train": os.path.join(base, f"{dataset_name}_train.csv"),
        "val": os.path.join(base, f"{dataset_name}_val.csv"),
        "test": os.path.join(base, f"{dataset_name}_test.csv"),
    }

    splits = {
        name: load_split(path)
        for name, path in files.items()
    }

    print("SPLIT SIZES")
    print("-" * 40)

    for name, samples in splits.items():
        print(f"{name:5}: {len(samples)}")

    print("\nCLASS DISTRIBUTION")
    print("-" * 40)

    for name, samples in splits.items():
        counts = Counter(sample["label"] for sample in samples)

        print(f"\n{name.upper()}")

        for label in sorted(counts):
            print(f"{label}: {counts[label]}")

    print("\nOVERLAP CHECK")
    print("-" * 40)

    path_sets = {
        name: set(sample["path"] for sample in samples)
        for name, samples in splits.items()
    }

    train_val = path_sets["train"] & path_sets["val"]
    train_test = path_sets["train"] & path_sets["test"]
    val_test = path_sets["val"] & path_sets["test"]

    print("Train ∩ Validation:", len(train_val))
    print("Train ∩ Test:      ", len(train_test))
    print("Validation ∩ Test: ", len(val_test))

    print("\nCLASS COVERAGE")
    print("-" * 40)

    class_sets = {
        name: set(sample["label"] for sample in samples)
        for name, samples in splits.items()
    }

    for name, classes in class_sets.items():
        print(f"{name:5}: {len(classes)} classes")

    print("\nPATH EXISTENCE CHECK")
    print("-" * 40)

    missing = 0

    for name, samples in splits.items():
        for sample in samples:
            if not resolve_image_path(sample["path"]).is_file():
                missing += 1

    print("Missing files:", missing)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:")
        print("python verify_split.py <dataset_name>")
        sys.exit(1)

    verify_split(sys.argv[1])