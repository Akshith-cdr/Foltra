"""Migrated build_tobrfv_temporal_index.py. Filename semantics remain unresolved; see docs/datasets/tobrfv.md."""
from src.utils.config import dataset_path, manifest_path, load_config, image_reference
from src.utils.preparation import run_preparation, check_outputs
import csv
import os
import re
from collections import defaultdict


BASE = str(dataset_path("tobrfv"))

MODALITIES = load_config()["datasets"]["tobrfv"]["modalities"]

OUTPUT = str(manifest_path("generated/tobrfv_temporal_index.csv"))


# ------------------------------------------------------------
# Filename format
#
# H_11_RGB_0_1.tiff
#
# H       = Healthy
# 11      = Day
# RGB     = Modality
# 0       = Camera angle
# 1       = Pot ID
#
# V_4_VNIR_800nm_5_12.png
#
# V       = Virus
# 4       = Day
# VNIR... = Modality
# 5       = Camera angle
# 12      = Pot ID
# ------------------------------------------------------------

PATTERN = re.compile(
    r"^(H|V)_(\d+)_(RGB|VNIR|VNIR_800nm|VNIR_1000nm)_(\d+)_(\d+)\.[^.]+$",
    re.IGNORECASE,
)


def parse_filename(filename):

    match = PATTERN.match(filename)

    if not match:
        return None

    status = match.group(1).upper()
    day = int(match.group(2))
    modality = match.group(3)
    angle = int(match.group(4))
    if angle not in load_config("preprocessing")["tobrfv"]["allowed_angles"]:
        return None
    pot_id = int(match.group(5))

    return {
        "status": status,
        "day": day,
        "modality": modality,
        "angle": angle,
        "pot_id": pot_id,
    }


def collect_files():

    data = defaultdict(dict)

    for modality in MODALITIES:

        root_dir = os.path.join(
            BASE,
            modality
        )

        for root, _, files in os.walk(root_dir):

            group = os.path.relpath(
                root,
                root_dir
            ).replace("/", "\\")

            for filename in files:

                parsed = parse_filename(filename)

                if parsed is None:
                    continue

                key = (
                    group,
                    parsed["status"],
                    parsed["day"],
                    parsed["angle"],
                    parsed["pot_id"],
                )

                data[key][modality] = image_reference(os.path.join(root, filename))

    return data


def print_summary(data):

    print()
    print("TEMPORAL INDEX SUMMARY")
    print("=" * 75)

    print(
        f"Total multimodal observations: {len(data)}"
    )

    groups = defaultdict(int)
    pots = defaultdict(set)
    days = defaultdict(set)

    for key in data:

        group, status, day, angle, pot_id = key

        groups[group] += 1

        pots[group].add(pot_id)

        days[group].add(day)

    print()
    print("GROUPS")
    print("-" * 75)

    for group in sorted(groups):

        print(
            f"{group:<25} | "
            f"Observations: {groups[group]:4d} | "
            f"Pots: {len(pots[group]):2d} | "
            f"Days: {sorted(days[group])}"
        )


def check_modalities(data):

    print()
    print("MODALITY COMPLETENESS")
    print("-" * 75)

    for modality in MODALITIES:

        count = sum(
            1
            for values in data.values()
            if modality in values
        )

        print(
            f"{modality:<15}: {count}"
        )

    complete = []
    incomplete = []

    for key, values in data.items():

        if all(
            modality in values
            for modality in MODALITIES
        ):
            complete.append(key)

        else:
            incomplete.append(
                (key, set(MODALITIES) - set(values))
            )

    print()
    print(
        f"Fully multimodal observations: "
        f"{len(complete)}"
    )

    print(
        f"Incomplete observations: "
        f"{len(incomplete)}"
    )

    if incomplete:

        print()
        print("INCOMPLETE EXAMPLES")

        for key, missing in incomplete[:20]:

            print(
                f"{key} | Missing: "
                f"{sorted(missing)}"
            )

    return complete


def check_temporal_structure(data):

    print()
    print("TEMPORAL STRUCTURE")
    print("-" * 75)

    plants = defaultdict(list)

    for key in data:

        group, status, day, angle, pot_id = key

        plants[
            (group, status, pot_id)
        ].append(
            (day, angle)
        )

    print(
        f"Unique plant/pot sequences: "
        f"{len(plants)}"
    )

    day_counts = []

    for key, observations in plants.items():

        days = sorted(
            set(
                day
                for day, angle in observations
            )
        )

        day_counts.append(len(days))

    if day_counts:

        print(
            f"Minimum days/plant : "
            f"{min(day_counts)}"
        )

        print(
            f"Maximum days/plant : "
            f"{max(day_counts)}"
        )

        print(
            f"Average days/plant : "
            f"{sum(day_counts) / len(day_counts):.2f}"
        )

    print()
    print("EXAMPLE TEMPORAL SEQUENCES")
    print("-" * 75)

    shown = 0

    for key in sorted(plants):

        days = sorted(
            set(
                day
                for day, angle in plants[key]
            )
        )

        print()
        print(
            f"Group : {key[0]}"
        )

        print(
            f"Status: {key[1]}"
        )

        print(
            f"Pot   : {key[2]}"
        )

        print(
            f"Days  : {days}"
        )

        shown += 1

        if shown >= 10:
            break


def save_index(data):

    os.makedirs(
        os.path.dirname(OUTPUT),
        exist_ok=True
    )

    fieldnames = [
        "group",
        "status",
        "pot_id",
        "day",
        "angle",
        "rgb_path",
        "vnir_path",
        "vnir_800nm_path",
        "vnir_1000nm_path",
    ]

    rows = []

    for key in sorted(data):

        group, status, day, angle, pot_id = key

        values = data[key]

        rows.append({
            "group": group,
            "status": status,
            "pot_id": pot_id,
            "day": day,
            "angle": angle,
            "rgb_path": values.get("RGB", ""),
            "vnir_path": values.get("VNIR", ""),
            "vnir_800nm_path": values.get(
                "VNIR_800nm",
                ""
            ),
            "vnir_1000nm_path": values.get(
                "VNIR_1000nm",
                ""
            ),
        })

    with open(
        OUTPUT,
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

    print()
    print("OUTPUT")
    print("-" * 75)

    print(
        f"Saved {len(rows)} observations"
    )

    print(
        f"Output: {OUTPUT}"
    )


def main():
    check_outputs([OUTPUT])

    print(
        "TOBRFV-LMID TEMPORAL INDEX BUILDER"
    )

    print("=" * 75)

    print(
        "Filename interpretation:"
    )

    print(
        "Status_Day_Modality_Angle_PotID"
    )

    print()

    data = collect_files()

    print(
        f"Collected observations: "
        f"{len(data)}"
    )

    print_summary(data)

    complete = check_modalities(data)

    check_temporal_structure(data)

    # --------------------------------------------------------
    # Only save fully multimodal observations.
    # --------------------------------------------------------

    complete_data = {
        key: data[key]
        for key in complete
    }

    print()
    print(
        f"Using {len(complete_data)} "
        f"fully aligned multimodal observations."
    )

    save_index(
        complete_data
    )


if __name__ == "__main__":
    run_preparation(main, [OUTPUT], unvalidated=True)