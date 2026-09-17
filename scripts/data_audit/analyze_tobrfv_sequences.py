"""Migrated diagnostic; TOBRFV semantics are provisional. See docs/datasets/tobrfv.md."""
from src.utils.config import dataset_path, manifest_path, resolve_image_path
import os
import re
from collections import defaultdict, Counter


BASE = str(dataset_path("tobrfv") / "RGB")

PATTERN = re.compile(
    r"^(H|V)_(\d+)_RGB_(\d+)_(\d+)\.",
    re.IGNORECASE
)


def main():
    data = defaultdict(lambda: defaultdict(Counter))

    for root, _, files in os.walk(BASE):
        group = os.path.relpath(root, BASE)

        for filename in files:
            match = PATTERN.match(filename)

            if not match:
                continue

            plant_type = match.group(1)
            identifier = match.group(2)
            capture = match.group(3)
            frame = match.group(4)

            sequence_id = f"{plant_type}_{identifier}"

            data[group][sequence_id][capture] += 1

    print("TOBRFV-LMID SEQUENCE ANALYSIS")
    print("=" * 80)

    total_frames = 0
    total_sequences = 0

    for group in sorted(data):
        sequences = data[group]

        print()
        print(f"GROUP: {group}")
        print("-" * 80)

        group_frames = 0

        for sequence_id in sorted(
            sequences,
            key=lambda x: (x[0], int(x.split("_")[1]))
        ):
            captures = sequences[sequence_id]

            sequence_total = sum(captures.values())
            group_frames += sequence_total

            capture_text = ", ".join(
                f"{capture}: {count}"
                for capture, count in sorted(
                    captures.items(),
                    key=lambda x: int(x[0])
                )
            )

            print(
                f"  {sequence_id:<6} | "
                f"Frames: {sequence_total:<4} | "
                f"Captures: {capture_text}"
            )

        print("-" * 80)
        print(
            f"  Sequences: {len(sequences)} | "
            f"Total frames: {group_frames}"
        )

        total_sequences += len(sequences)
        total_frames += group_frames

    print()
    print("=" * 80)
    print("OVERALL")
    print("=" * 80)
    print(f"Groups:          {len(data)}")
    print(f"Sequences:       {total_sequences}")
    print(f"Total RGB frames:{total_frames}")


if __name__ == "__main__":
    main()