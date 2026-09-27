"""Load the preserved plantvillage classification split (PlantVillage uses color only)."""
from .classification import ClassificationDataset


def create_dataset(split="train", transform=None, class_to_idx=None):
    if split not in ("train", "val", "test"):
        raise ValueError("Unknown split")
    return ClassificationDataset(f"plantvillage_{split}.csv", transform, class_to_idx)
