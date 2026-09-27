"""Dataset-specific statistics; explicitly invoked, never run on import."""
from src.utils.config import dataset_path

def main():
    import os
    import json
    from collections import Counter

    base = str(dataset_path("plantdoc"))

    image_counts = Counter()
    object_counts = Counter()

    for split in ["train", "test"]:
        ann_dir = os.path.join(base, split, "ann")

        for filename in os.listdir(ann_dir):
            if not filename.endswith(".json"):
                continue

            path = os.path.join(ann_dir, filename)

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            labels_in_image = set()

            for obj in data.get("objects", []):
                label = obj.get("classTitle")

                if label:
                    object_counts[label] += 1
                    labels_in_image.add(label)

            for label in labels_in_image:
                image_counts[label] += 1

    print("CLASS | IMAGES | OBJECTS")
    print("-" * 60)

    for label in sorted(object_counts):
        print(
            f"{label} | "
            f"{image_counts[label]} | "
            f"{object_counts[label]}"
        )

if __name__ == "__main__":
    main()
