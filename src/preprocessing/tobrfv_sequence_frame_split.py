"""Migrated split_tobrfv_sequences.py. Filename semantics remain unresolved; see docs/datasets/tobrfv.md."""
from src.utils.config import dataset_path, manifest_path, load_config, image_reference
from src.utils.preparation import run_preparation, check_outputs
import csv
import os
import random
import re
from collections import defaultdict, Counter


BASE = str(dataset_path("tobrfv"))
RGB_DIR = os.path.join(BASE, "RGB")

OUTPUT_DIR = str(manifest_path("generated"))

RANDOM_SEED = load_config("preprocessing")["random_seed"]

TRAIN_RATIO = load_config("preprocessing")["tobrfv"]["train_ratio"]
VAL_RATIO = load_config("preprocessing")["tobrfv"]["val_ratio"]
TEST_RATIO = load_config("preprocessing")["tobrfv"]["test_ratio"]

PATTERN = re.compile(
    r"^(H|V)_(\d+)_RGB_(\d+)_(\d+)\.(.+)$",
    re.IGNORECASE
)


def collect_sequences():
    """
    Collect RGB frames and organize them by:

        (group, sequence_id)

    Example:
        ('Tomato\\I\\Virus', 'V_20')
    """

    sequences = defaultdict(list)

    for root, _, files in os.walk(RGB_DIR):

        group = os.path.relpath(root, RGB_DIR).replace("/", "\\")

        for filename in files:

            match = PATTERN.match(filename)

            if not match:
                continue

            plant_type = match.group(1).upper()
            identifier = match.group(2)
            capture = match.group(3)
            frame = match.group(4)

            sequence_id = f"{plant_type}_{identifier}"

            rgb_path = image_reference(os.path.join(root, filename))

            sequences[(group, sequence_id)].append({
                "group": group,
                "sequence_id": sequence_id,
                "plant_type": (
                    "Healthy"
                    if plant_type == "H"
                    else "Virus"
                ),
                "identifier": identifier,
                "capture": int(capture),
                "frame": int(frame),
                "rgb_path": rgb_path,
            })

    return sequences


def validate_ratios():

    total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO

    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0"
        )


def split_group_sequences(group_sequences):

    items = list(group_sequences)

    random.shuffle(items)

    total = len(items)

    train_count = round(total * TRAIN_RATIO)
    val_count = round(total * VAL_RATIO)

    if train_count + val_count > total:
        val_count = total - train_count

    train = items[:train_count]

    val = items[
        train_count:
        train_count + val_count
    ]

    test = items[
        train_count + val_count:
    ]

    return train, val, test


def create_rows(sequence_keys, sequences, split):

    """
    Convert sequence keys into frame-level CSV rows.
    """

    rows = []

    for key in sequence_keys:

        frames = sequences[key]

        for item in frames:

            rows.append({
                "split": split,
                "group": item["group"],
                "plant_type": item["plant_type"],
                "sequence_id": item["sequence_id"],
                "identifier": item["identifier"],
                "capture": item["capture"],
                "frame": item["frame"],
                "rgb_path": item["rgb_path"],
            })

    return rows


def save_csv(rows, filename):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    fieldnames = [
        "split",
        "group",
        "plant_type",
        "sequence_id",
        "identifier",
        "capture",
        "frame",
        "rgb_path",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    return output_path


def print_sequence_summary(split_sequences):

    print()
    print("SEQUENCE DISTRIBUTION")
    print("-" * 75)

    for split in ["train", "val", "test"]:

        groups = Counter()

        for group, sequence_id in split_sequences[split]:

            groups[group] += 1

        print()
        print(split.upper())

        for group in sorted(groups):

            print(
                f"{group:<25} : "
                f"{groups[group]} sequences"
            )

        print(
            f"TOTAL SEQUENCES: "
            f"{sum(groups.values())}"
        )


def print_frame_summary(rows):

    print()
    print("FRAME DISTRIBUTION")
    print("-" * 75)

    for split in ["train", "val", "test"]:

        split_rows = [
            row for row in rows
            if row["split"] == split
        ]

        print(
            f"{split:<5}: "
            f"{len(split_rows)} frames"
        )


def verify_no_sequence_leakage(split_sequences):

    sets = {
        split: set(split_sequences[split])
        for split in ["train", "val", "test"]
    }

    train_val = sets["train"] & sets["val"]

    train_test = sets["train"] & sets["test"]

    val_test = sets["val"] & sets["test"]

    print()
    print("SEQUENCE LEAKAGE CHECK")
    print("-" * 75)

    print(
        f"Train âˆ© Validation: {len(train_val)}"
    )

    print(
        f"Train âˆ© Test:       {len(train_test)}"
    )

    print(
        f"Validation âˆ© Test:  {len(val_test)}"
    )

    if train_val or train_test or val_test:

        raise RuntimeError(
            "Sequence leakage detected!"
        )

    print(
        "PASS: No sequence appears "
        "in multiple splits."
    )


def main():
    check_outputs([os.path.join(OUTPUT_DIR, "tobrfv_" + split + ".csv") for split in ("train", "val", "test")])

    validate_ratios()

    random.seed(RANDOM_SEED)

    print(
        "TOBRFV-LMID SEQUENCE-LEVEL SPLITTER"
    )

    print("=" * 75)

    print(
        f"Random seed: {RANDOM_SEED}"
    )

    print(
        f"Target split: "
        f"{TRAIN_RATIO:.0%} / "
        f"{VAL_RATIO:.0%} / "
        f"{TEST_RATIO:.0%}"
    )

    print()
    print("Collecting RGB sequences...")

    sequences = collect_sequences()

    print(
        f"Found {len(sequences)} sequences."
    )

    # ---------------------------------------------------------------
    # Organize sequences by experimental group
    # ---------------------------------------------------------------

    group_to_sequences = defaultdict(list)

    for key in sequences:

        group_to_sequences[key[0]].append(key)

    split_sequences = {
        "train": [],
        "val": [],
        "test": [],
    }

    print()
    print("SPLITTING GROUPS")
    print("-" * 75)

    for group in sorted(group_to_sequences):

        group_items = group_to_sequences[group]

        train, val, test = split_group_sequences(
            group_items
        )

        split_sequences["train"].extend(train)

        split_sequences["val"].extend(val)

        split_sequences["test"].extend(test)

        print(
            f"{group:<25} | "
            f"Train: {len(train):2d} | "
            f"Val: {len(val):2d} | "
            f"Test: {len(test):2d}"
        )

    # ---------------------------------------------------------------
    # Check leakage
    # ---------------------------------------------------------------

    verify_no_sequence_leakage(
        split_sequences
    )

    # ---------------------------------------------------------------
    # Convert sequences to frame-level rows
    # ---------------------------------------------------------------

    train_rows = create_rows(
        split_sequences["train"],
        sequences,
        "train"
    )

    val_rows = create_rows(
        split_sequences["val"],
        sequences,
        "val"
    )

    test_rows = create_rows(
        split_sequences["test"],
        sequences,
        "test"
    )

    all_rows = (
        train_rows +
        val_rows +
        test_rows
    )

    # ---------------------------------------------------------------
    # Save CSV files
    # ---------------------------------------------------------------

    train_path = save_csv(
        train_rows,
        "tobrfv_train.csv"
    )

    val_path = save_csv(
        val_rows,
        "tobrfv_val.csv"
    )

    test_path = save_csv(
        test_rows,
        "tobrfv_test.csv"
    )

    # ---------------------------------------------------------------
    # Summaries
    # ---------------------------------------------------------------

    print_sequence_summary(
        split_sequences
    )

    print_frame_summary(
        all_rows
    )

    print()
    print("OUTPUT FILES")
    print("-" * 75)

    print(train_path)

    print(val_path)

    print(test_path)

    print()
    print("=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == "__main__":
    run_preparation(main, [os.path.join(OUTPUT_DIR, "tobrfv_" + split + ".csv") for split in ("train", "val", "test")], unvalidated=True)