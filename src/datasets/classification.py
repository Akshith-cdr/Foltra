"""Manifest-based classification loading for prepared single-label datasets."""
from PIL import Image
from torch.utils.data import Dataset
from src.utils.manifests import load_manifest
from src.utils.config import resolve_image_path


class ClassificationDataset(Dataset):
    def __init__(self, manifest, transform=None, class_to_idx=None):
        self.rows = load_manifest(manifest, required=("path", "label"))
        self.transform = transform
        labels = sorted({row["label"] for row in self.rows})
        self.class_to_idx = class_to_idx if class_to_idx is not None else {label: i for i, label in enumerate(labels)}
        if not set(labels).issubset(self.class_to_idx):
            raise ValueError("Manifest contains labels outside the supplied class mapping")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(resolve_image_path(row["path"])) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_idx[row["label"]]
