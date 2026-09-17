"""Migrated diagnostic; TOBRFV semantics are provisional. See docs/datasets/tobrfv.md."""
from src.utils.config import dataset_path, manifest_path, resolve_image_path
import csv
import os
import re
from collections import defaultdict


CSV_PATH = str(manifest_path("tobrfv_train.csv"))

FRAME_PATTERN = re.compile(
    r"^(H|V)_(\d+)_RGB_(\d+)_(\d+)\.tiff$",
    re.IGNORECASE
)


def load_rows():

    rows = []

    with open(
        CSV_PATH,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def analyze():

    print("TOBRFV-LMID TEMPORAL STRUCTURE VERIFICATION")
    print("=" * 75)

    rows = load_rows()

    print(f"Rows loaded: {len(rows)}")

    # ---------------------------------------------------------------
    # Group frames by sequence
    # ---------------------------------------------------------------

    sequences = defaultdict(list)

    for row in rows:

        key = (
            row["group"],
            row["sequence_id"]
        )

        sequences[key].append(row)

    print(
        f"Unique sequences: {len(sequences)}"
    )

    # ---------------------------------------------------------------
    # Analyze frame ordering
    # ---------------------------------------------------------------

    print()
    print("FRAME INDEX ANALYSIS")
    print("-" * 75)

    total_sequences = len(sequences)
    ordered_sequences = 0
    duplicate_frames = 0
    missing_indices = 0

    frame_counts = []

    for key, sequence_rows in sorted(sequences.items()):

        indices = []

        for row in sequence_rows:
            indices.append(int(row["frame"]))

        indices_sorted = sorted(indices)

        frame_counts.append(len(indices_sorted))

        if indices != indices_sorted:
            # CSV order is not necessarily numerical order.
            # This is not an error; we check the actual indices below.
            pass

        if len(indices_sorted) != len(set(indices_sorted)):
            duplicate_frames += 1

        if len(indices_sorted) > 1:

            expected = set(
                range(
                    min(indices_sorted),
                    max(indices_sorted) + 1
                )
            )

            actual = set(indices_sorted)

            missing = expected - actual

            if not missing:
                ordered_sequences += 1
            else:
                missing_indices += 1

    print(
        f"Sequences checked        : {total_sequences}"
    )

    print(
        f"Sequences with continuous "
        f"frame indices             : {ordered_sequences}"
    )

    print(
        f"Sequences with missing "
        f"frame indices             : {missing_indices}"
    )

    print(
        f"Sequences with duplicate "
        f"frame indices             : {duplicate_frames}"
    )

    # ---------------------------------------------------------------
    # Frame count statistics
    # ---------------------------------------------------------------

    print()
    print("FRAME COUNT STATISTICS")
    print("-" * 75)

    print(
        f"Minimum frames/sequence : {min(frame_counts)}"
    )

    print(
        f"Maximum frames/sequence : {max(frame_counts)}"
    )

    print(
        f"Average frames/sequence : "
        f"{sum(frame_counts) / len(frame_counts):.2f}"
    )

    # ---------------------------------------------------------------
    # Show representative sequences
    # ---------------------------------------------------------------

    print()
    print("REPRESENTATIVE SEQUENCES")
    print("-" * 75)

    shown = 0

    for key, sequence_rows in sorted(sequences.items()):

        indices = sorted(
            int(row["frame"])
            for row in sequence_rows
        )

        print()
        print(
            f"Group    : {key[0]}"
        )

        print(
            f"Sequence : {key[1]}"
        )

        print(
            f"Frames   : {len(indices)}"
        )

        print(
            f"Indices  : {indices}"
        )

        shown += 1

        if shown >= 10:
            break

    # ---------------------------------------------------------------
    # Check capture indices
    # ---------------------------------------------------------------

    print()
    print("CAPTURE INDEX ANALYSIS")
    print("-" * 75)

    capture_counts = defaultdict(set)

    for row in rows:

        key = (
            row["group"],
            row["sequence_id"]
        )

        capture_counts[key].add(
            row["capture"]
        )

    capture_distribution = defaultdict(int)

    for captures in capture_counts.values():

        capture_distribution[
            tuple(sorted(captures))
        ] += 1

    for captures, count in sorted(
        capture_distribution.items()
    ):

        print(
            f"Capture indices {captures}: "
            f"{count} sequences"
        )

    # ---------------------------------------------------------------
    # Important interpretation
    # ---------------------------------------------------------------

    print()
    print("=" * 75)
    print("INTERPRETATION")
    print("=" * 75)

    print(
        "This script verifies filename/index structure only."
    )

    print(
        "It does NOT prove that frame indices represent "
        "real temporal progression."
    )

    print()
    print(
        "To establish temporal validity, the dataset documentation "
        "or acquisition protocol must confirm what the final frame "
        "index represents."
    )


if __name__ == "__main__":
    analyze()