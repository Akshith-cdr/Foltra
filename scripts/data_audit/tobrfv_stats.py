"""Dataset-specific statistics; explicitly invoked, never run on import."""
from src.utils.config import dataset_path

def main():
    import os
    from collections import Counter

    base = str(dataset_path("tobrfv"))

    for modality in sorted(os.listdir(base)):
        modality_path = os.path.join(base, modality)

        if not os.path.isdir(modality_path):
            continue

        print("\n==========", modality, "==========")

        counts = Counter()

        for plant in os.listdir(modality_path):
            plant_path = os.path.join(modality_path, plant)

            if not os.path.isdir(plant_path):
                continue

            for group in os.listdir(plant_path):
                group_path = os.path.join(plant_path, group)

                if not os.path.isdir(group_path):
                    continue

                for label in os.listdir(group_path):
                    label_path = os.path.join(group_path, label)

                    if not os.path.isdir(label_path):
                        continue

                    count = sum(
                        1
                        for f in os.listdir(label_path)
                        if os.path.isfile(os.path.join(label_path, f))
                    )

                    counts[(plant, group, label)] = count

        for key, count in sorted(counts.items()):
            plant, group, label = key
            print(f"{plant:8} | {group:2} | {label:7} | {count}")

if __name__ == "__main__":
    main()
