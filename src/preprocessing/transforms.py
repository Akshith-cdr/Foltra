"""Classification transforms; do not apply random transforms independently to aligned modalities."""
from io import BytesIO
import random

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from src.utils.config import load_config


def to_rgb(image):
    return image.convert("RGB")


class RandomJpegCompression:
    """Mild PIL JPEG round-trip for training images only."""

    def __init__(self, probability, quality):
        self.probability = float(probability)
        self.quality = tuple(int(value) for value in quality)
        if not 0 <= self.probability <= 1:
            raise ValueError("JPEG probability must be between zero and one")
        if len(self.quality) != 2 or not 1 <= self.quality[0] <= self.quality[1] <= 100:
            raise ValueError("JPEG quality must be an inclusive [minimum, maximum] pair")

    def __call__(self, image):
        if random.random() >= self.probability:
            return image
        quality = random.randint(*self.quality)
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")

    def __repr__(self):
        return (f"{self.__class__.__name__}(probability={self.probability}, "
                f"quality={self.quality})")


def build_field_style_train_transforms(config, image_config=None):
    """Build the controlled Week 6 field-style training pipeline."""
    cfg = image_config or load_config("preprocessing")["images"]
    if config.get("profile") != "field_style_v1":
        raise ValueError("Expected augmentation profile field_style_v1")
    crop = config["random_resized_crop"]
    jitter = config["color_jitter"]
    blur = config["gaussian_blur"]
    jpeg = config["jpeg_compression"]
    return transforms.Compose([
        transforms.Lambda(to_rgb),
        transforms.RandomResizedCrop(
            cfg["size"], scale=tuple(crop["scale"]), ratio=tuple(crop["ratio"]),
            interpolation=InterpolationMode.BILINEAR, antialias=True,
        ),
        transforms.RandomHorizontalFlip(float(config["horizontal_flip_probability"])),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=float(jitter["brightness"]), contrast=float(jitter["contrast"]),
                saturation=float(jitter["saturation"]), hue=float(jitter["hue"]),
            )
        ], p=float(jitter["probability"])),
        transforms.RandomApply([
            transforms.GaussianBlur(int(blur["kernel_size"]), sigma=tuple(blur["sigma"]))
        ], p=float(blur["probability"])),
        RandomJpegCompression(jpeg["probability"], jpeg["quality"]),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])


def build_transforms(training=False, augmentation=None):
    cfg = load_config("preprocessing")["images"]
    if training and augmentation is not None:
        return build_field_style_train_transforms(augmentation, cfg)
    steps = [transforms.Lambda(to_rgb), transforms.Resize((cfg["size"], cfg["size"]))]
    if training:
        steps += [transforms.RandomHorizontalFlip(cfg["horizontal_flip_probability"]),
                  transforms.RandomRotation(cfg["rotation_degrees"])]
    steps += [transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"])]
    return transforms.Compose(steps)


train_transforms = build_transforms(True)
val_test_transforms = build_transforms(False)
