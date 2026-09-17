"""Classification transforms; do not apply random transforms independently to aligned modalities."""
from torchvision import transforms
from src.utils.config import load_config


def to_rgb(image):
    return image.convert("RGB")


def build_transforms(training=False):
    cfg = load_config("preprocessing")["images"]
    steps = [transforms.Lambda(to_rgb), transforms.Resize((cfg["size"], cfg["size"]))]
    if training:
        steps += [transforms.RandomHorizontalFlip(cfg["horizontal_flip_probability"]),
                  transforms.RandomRotation(cfg["rotation_degrees"])]
    steps += [transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"])]
    return transforms.Compose(steps)


train_transforms = build_transforms(True)
val_test_transforms = build_transforms(False)
