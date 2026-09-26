"""Check transforms against generated in-memory pixels; no raw-data reads."""
from PIL import Image
from src.preprocessing.transforms import train_transforms, val_test_transforms


def main():
    image = Image.new("RGB", (32, 24))
    for transform in (train_transforms, val_test_transforms):
        tensor = transform(image)
        print(tuple(tensor.shape), tensor.dtype)


if __name__ == "__main__":
    main()
