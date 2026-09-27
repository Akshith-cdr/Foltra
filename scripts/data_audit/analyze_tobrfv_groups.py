"""Migrated diagnostic; TOBRFV semantics are provisional. See docs/datasets/tobrfv.md."""
from src.utils.config import dataset_path, manifest_path, resolve_image_path
import csv
from collections import Counter, defaultdict
from pathlib import Path


INDEX = manifest_path("tobrfv_index.csv")


def main():
    print("CAUTION: the historical plant column is crop species, not biological plant identity.")

    if not INDEX.exists():
        print(f"ERROR: Index not found: {INDEX}")
        return

    rows = []

    with open(INDEX, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print("TOBRFV-LMID GROUP / LEAKAGE ANALYSIS")
    print("=" * 70)

    print(f"Total aligned observations: {len(rows)}")

    # ---------------------------------------------------------
    # Basic checks
    # ---------------------------------------------------------

    required_columns = [
        "sample_id",
        "group",
        "label",
        "plant",
        "experiment",
        "rgb_path",
        "vnir_path",
        "vnir_800nm_path",
        "vnir_1000nm_path",
    ]

    missing_columns = [
        c for c in required_columns
        if c not in rows[0]
    ]

    if missing_columns:
        print()
        print("ERROR: Missing columns:")
        for c in missing_columns:
            print(f"  {c}")
        return

    # ---------------------------------------------------------
    # Check duplicate observations
    # ---------------------------------------------------------

    sample_ids = [r["sample_id"] for r in rows]
    groups = [r["group"] for r in rows]

    print()
    print("DUPLICATE CHECK")
    print("-" * 70)

    duplicate_samples = [
        sample_id
        for sample_id, count in Counter(sample_ids).items()
        if count > 1
    ]

    print(
        f"Duplicate sample IDs: "
        f"{len(duplicate_samples)}"
    )

    if duplicate_samples:
        for sample_id in duplicate_samples[:20]:
            print(f"  {sample_id}")

    # ---------------------------------------------------------
    # Group distribution
    # ---------------------------------------------------------

    group_counts = Counter(groups)

    print()
    print("GROUP DISTRIBUTION")
    print("-" * 70)

    for group, count in sorted(group_counts.items()):
        print(f"{group:<35} {count}")

    print()
    print(f"Number of groups: {len(group_counts)}")

    # ---------------------------------------------------------
    # Plant-level distribution
    # ---------------------------------------------------------

    plant_groups = defaultdict(set)

    for row in rows:
        plant_groups[row["plant"]].add(row["group"])

    print()
    print("PLANT IDENTIFIERS")
    print("-" * 70)

    print(
        f"Unique plant/sample identifiers: "
        f"{len(plant_groups)}"
    )

    for plant, group_set in sorted(plant_groups.items()):
        print(
            f"{plant:<10} -> "
            f"{len(group_set)} group(s): "
            f"{', '.join(sorted(group_set))}"
        )

    # ---------------------------------------------------------
    # Check whether a plant identifier appears across groups
    # ---------------------------------------------------------

    cross_group_plants = {
        plant: group_set
        for plant, group_set in plant_groups.items()
        if len(group_set) > 1
    }

    print()
    print("CROSS-GROUP PLANT CHECK")
    print("-" * 70)

    print(
        f"Plant IDs appearing in multiple groups: "
        f"{len(cross_group_plants)}"
    )

    if cross_group_plants:

        for plant, group_set in sorted(
            cross_group_plants.items()
        ):

            print(
                f"{plant}: "
                f"{', '.join(sorted(group_set))}"
            )

    # ---------------------------------------------------------
    # Observation counts per plant
    # ---------------------------------------------------------

    plant_counts = Counter(
        row["plant"]
        for row in rows
    )

    print()
    print("OBSERVATIONS PER PLANT ID")
    print("-" * 70)

    for plant, count in sorted(plant_counts.items()):
        print(f"{plant:<10} {count}")

    # ---------------------------------------------------------
    # Check modality paths
    # ---------------------------------------------------------

    print()
    print("MODALITY PATH CHECK")
    print("-" * 70)

    modality_columns = [
        "rgb_path",
        "vnir_path",
        "vnir_800nm_path",
        "vnir_1000nm_path",
    ]

    missing_paths = []

    for row in rows:

        for column in modality_columns:

            path = resolve_image_path(row[column])

            if not path.exists():
                missing_paths.append(
                    (row["group"], row["sample_id"], column, path)
                )

    print(
        f"Missing modality files: "
        f"{len(missing_paths)}"
    )

    if missing_paths:

        for group, sample_id, modality, path in missing_paths[:20]:
            print(
                f"  {group} | {sample_id} | "
                f"{modality} | {path}"
            )

    # ---------------------------------------------------------
    # Final recommendation
    # ---------------------------------------------------------

    print()
    print("=" * 70)

    if cross_group_plants:

        print(
            "RECOMMENDATION: "
            "Use group-aware splitting."
        )

        print(
            "Do NOT perform a random image-level split."
        )

    else:

        print(
            "RECOMMENDATION: "
            "Plant-level grouping can be used for splitting."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()