"""Migrated diagnostic; TOBRFV semantics are provisional. See docs/datasets/tobrfv.md."""
from src.utils.config import dataset_path, manifest_path, resolve_image_path
import os
import re
from collections import defaultdict


BASE = str(dataset_path("tobrfv") / "RGB")

PATTERN = re.compile(
    r"^(H|V)_(\d+)_RGB_(\d+)_(\d+)\.",
    re.IGNORECASE
)


def main():
    identifiers = defaultdict(lambda: defaultdict(set))

    for root, _, files in os.walk(BASE):
        group = os.path.relpath(root, BASE)

        for filename in files:
            match = PATTERN.match(filename)

            if not match:
                continue

            plant_type = match.group(1).upper()
            identifier = match.group(2)

            identifiers[plant_type][identifier].add(group)

    print("TOBRFV-LMID IDENTIFIER STRUCTURE")
    print("=" * 75)

    for plant_type in sorted(identifiers):
        print()
        print(f"{plant_type} IDENTIFIERS")
        print("-" * 75)

        for identifier in sorted(
            identifiers[plant_type],
            key=lambda x: int(x)
        ):
            groups = sorted(identifiers[plant_type][identifier])

            print(
                f"{plant_type}_{identifier:<3} | "
                f"{len(groups)} groups | "
                f"{', '.join(groups)}"
            )

    print()
    print("=" * 75)
    print("SUMMARY")
    print("=" * 75)

    for plant_type in sorted(identifiers):
        ids = identifiers[plant_type]

        print(
            f"{plant_type}: "
            f"{len(ids)} unique identifiers"
        )

        counts = defaultdict(int)

        for identifier, groups in ids.items():
            counts[len(groups)] += 1

        for group_count in sorted(counts):
            print(
                f"  Appearing in {group_count} group(s): "
                f"{counts[group_count]} identifiers"
            )

    print()
    print("=" * 75)
    print("IMPORTANT")
    print("=" * 75)
    print(
        "This script only reports filename/group relationships."
    )
    print(
        "It does NOT assume that identical identifiers represent "
        "the same biological plant."
    )


if __name__ == "__main__":
    main()