"""Migrated build_tobrfv_index.py. Filename semantics remain unresolved; see docs/datasets/tobrfv.md."""
from src.utils.config import dataset_path, manifest_path, load_config, image_reference
from src.utils.preparation import run_preparation, check_outputs
import csv
import re
from pathlib import Path


BASE = dataset_path("tobrfv")
OUTPUT = manifest_path("generated/tobrfv_index.csv")


MODALITIES = {
    "RGB": re.compile(
        r"^(H|V)_(\d+)_RGB_(.+)\.tiff$",
        re.IGNORECASE,
    ),

    "VNIR": re.compile(
        r"^(H|V)_(\d+)_VNIR_(.+)\.png$",
        re.IGNORECASE,
    ),

    "VNIR_800nm": re.compile(
        r"^(H|V)_(\d+)_VNIR_800nm_(.+)\.png$",
        re.IGNORECASE,
    ),

    "VNIR_1000nm": re.compile(
        r"^(H|V)_(\d+)_VNIR_1000nm_(.+)\.png$",
        re.IGNORECASE,
    ),
}


# Modality selection is centralized; filename grammars remain dataset-specific.
MODALITIES = {name: MODALITIES[name] for name in load_config()["datasets"]["tobrfv"]["modalities"]}


def build_index(modality):
    """
    Build:

        (group, sample_id) -> file path
    """

    index = {}
    root = BASE / modality
    pattern = MODALITIES[modality]

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        match = pattern.match(path.name)

        if not match:
            continue

        prefix = match.group(1)
        plant_id = match.group(2)
        capture_id = match.group(3)

        sample_id = f"{prefix}_{plant_id}_{capture_id}"

        group = str(path.parent.relative_to(root)).replace("/", "\\")

        key = (group, sample_id)

        index[key] = path

    return index


def main():
    check_outputs([OUTPUT])

    print("Building TOBRFV-LMID multimodal index...")
    print("=" * 60)

    indexes = {}

    # ---------------------------------------------------------
    # Build modality indexes
    # ---------------------------------------------------------

    for modality in MODALITIES:

        indexes[modality] = build_index(modality)

        print(
            f"{modality:<15}: "
            f"{len(indexes[modality])}"
        )

    print("=" * 60)

    # ---------------------------------------------------------
    # RGB is the reference modality
    # ---------------------------------------------------------

    reference = set(indexes["RGB"].keys())

    print(
        f"RGB reference samples: "
        f"{len(reference)}"
    )

    # ---------------------------------------------------------
    # Find fully aligned samples
    # ---------------------------------------------------------

    aligned = reference.copy()

    for modality in [
        "VNIR",
        "VNIR_800nm",
        "VNIR_1000nm",
    ]:

        aligned &= set(indexes[modality].keys())

    # ---------------------------------------------------------
    # Find excluded RGB samples
    # ---------------------------------------------------------

    excluded = reference - aligned

    print()

    print(
        f"Fully aligned samples : "
        f"{len(aligned)}"
    )

    print(
        f"Excluded samples      : "
        f"{len(excluded)}"
    )

    if excluded:

        print()
        print("Excluded RGB samples:")

        for group, sample_id in sorted(excluded):

            print(
                f"  {group} | {sample_id}"
            )

    # ---------------------------------------------------------
    # Check extra samples in each modality
    # ---------------------------------------------------------

    print()
    print("Extra samples:")
    print("-" * 60)

    for modality in [
        "VNIR",
        "VNIR_800nm",
        "VNIR_1000nm",
    ]:

        extra = (
            set(indexes[modality].keys())
            - reference
        )

        print(
            f"{modality:<15}: "
            f"{len(extra)}"
        )

        for group, sample_id in sorted(extra):

            print(
                f"  {group} | {sample_id}"
            )

    # ---------------------------------------------------------
    # Build aligned rows
    # ---------------------------------------------------------

    rows = []

    for group, sample_id in sorted(aligned):

        rows.append({

            "sample_id": sample_id,

            "group": group,

            "label": group.split("\\")[-1],

            "plant": group.split("\\")[0],

            "experiment": group.split("\\")[1],

            "rgb_path": image_reference(
                indexes["RGB"][
                    (group, sample_id)
                ]
            ),

            "vnir_path": image_reference(
                indexes["VNIR"][
                    (group, sample_id)
                ]
            ),

            "vnir_800nm_path": image_reference(
                indexes["VNIR_800nm"][
                    (group, sample_id)
                ]
            ),

            "vnir_1000nm_path": image_reference(
                indexes["VNIR_1000nm"][
                    (group, sample_id)
                ]
            ),
        })

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
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

    with open(
        OUTPUT,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print()
    print("=" * 60)

    print(
        f"Saved {len(rows)} aligned samples"
    )

    print(
        f"Output: {OUTPUT}"
    )

    print("=" * 60)


if __name__ == "__main__":
    run_preparation(main, [OUTPUT], unvalidated=False)