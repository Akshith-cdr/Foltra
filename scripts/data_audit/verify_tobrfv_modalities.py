"""Migrated diagnostic; TOBRFV semantics are provisional. See docs/datasets/tobrfv.md."""
from src.utils.config import dataset_path, manifest_path, resolve_image_path
import os
import re


BASE = str(dataset_path("tobrfv"))

MODALITIES = [
    "RGB",
    "VNIR",
    "VNIR_800nm",
    "VNIR_1000nm",
]


def normalize_filename(filename, modality):
    """
    Convert a modality-specific filename into a common sample key.

    Example:
        H_11_RGB_0_1.tiff
        H_11_VNIR_0_1.png
        H_11_VNIR_800nm_0_1.png
        H_11_VNIR_1000nm_0_1.png

    All become:
        H_11_0_1
    """

    name, _ = os.path.splitext(filename)

    pattern = (
        r"^(H|V)_(\d+)_"
        + re.escape(modality)
        + r"_(.*)$"
    )

    match = re.match(pattern, name, re.IGNORECASE)

    if match is None:
        return None

    prefix = match.group(1)
    sample_id = match.group(2)
    observation = match.group(3)

    return f"{prefix}_{sample_id}_{observation}"


def build_index(modality):
    """
    Build an index of all files for one modality.

    Key:
        (relative_group_path, normalized_sample_key)

    Value:
        actual file path
    """

    modality_path = os.path.join(BASE, modality)

    index = {}

    for root, _, files in os.walk(modality_path):

        relative_group = os.path.relpath(
            root,
            modality_path
        )

        for filename in files:

            key = normalize_filename(
                filename,
                modality
            )

            if key is None:
                continue

            full_key = (
                relative_group,
                key
            )

            index[full_key] = os.path.join(
                root,
                filename
            )

    return index


# ---------------------------------------------------------
# BUILD MODALITY INDICES
# ---------------------------------------------------------

def main():
    indexes = {}

    for modality in MODALITIES:
        indexes[modality] = build_index(modality)


    # ---------------------------------------------------------
    # MODALITY COUNTS
    # ---------------------------------------------------------

    print("MODALITY COUNTS")
    print("-" * 50)

    for modality in MODALITIES:
        print(
            f"{modality:15}: "
            f"{len(indexes[modality])}"
        )


    # ---------------------------------------------------------
    # CORRESPONDENCE CHECK
    # ---------------------------------------------------------

    print("\nCORRESPONDENCE CHECK")
    print("-" * 50)

    reference = indexes["RGB"]

    reference_keys = set(
        reference.keys()
    )

    for modality in MODALITIES[1:]:

        modality_keys = set(
            indexes[modality].keys()
        )

        missing = reference_keys - modality_keys
        extra = modality_keys - reference_keys

        print(f"\nRGB -> {modality}")

        print(
            "Missing:",
            len(missing)
        )

        print(
            "Extra:  ",
            len(extra)
        )

        if missing:

            print("Missing examples:")

            for item in sorted(missing)[:10]:
                print(" ", item)

        if extra:

            print("Extra examples:")

            for item in sorted(extra)[:10]:
                print(" ", item)


    # ---------------------------------------------------------
    # GROUP COUNTS
    # ---------------------------------------------------------

    print("\nGROUP COUNTS")
    print("-" * 50)

    for modality in MODALITIES:

        groups = set(
            key[0]
            for key in indexes[modality].keys()
        )

        print(
            f"{modality:15}: "
            f"{len(groups)} groups"
        )


    # ---------------------------------------------------------
    # EXAMPLE MATCHES
    # ---------------------------------------------------------

    print("\nEXAMPLE MATCH")
    print("-" * 50)

    for key in sorted(reference_keys)[:5]:

        print("\nGroup :", key[0])
        print("Sample:", key[1])

        for modality in MODALITIES:

            path = indexes[modality].get(key)

            if path:
                print(
                    f"{modality:15}: "
                    f"{path}"
                )
            else:
                print(
                    f"{modality:15}: "
                    f"MISSING"
                )

if __name__ == "__main__":
    main()
