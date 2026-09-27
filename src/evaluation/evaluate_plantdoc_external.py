"""Evaluate the frozen Week 5 PlantVillage classifier on PlantDoc object crops."""
import argparse
import csv
import hashlib
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import sklearn
import torch
import torchvision
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

from src.models.resnet_baseline import build_model
from src.preprocessing.transforms import build_transforms
from src.training.train_baseline import select_device
from src.utils.config import dataset_path, ensure_output_path, image_reference, load_config, project_path
from src.utils.progress import progress_batches


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_collection(paths, root):
    """Hash file names and contents in a deterministic order."""
    digest = hashlib.sha256()
    for path in sorted((Path(item) for item in paths), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def resolve_plantdoc_root(config):
    """Resolve PlantDoc through the configured dataset root, including environment overrides."""
    return dataset_path(config["dataset"])


class PlantDocObjectDataset(Dataset):
    """Approved PlantDoc rectangle annotations, exposed as classification crops."""

    def __init__(self, root, split, label_mapping, class_to_idx, transform):
        if split != "test":
            raise ValueError("The first external-domain evaluation is restricted to the PlantDoc test split")
        self.root = Path(root)
        self.split = split
        self.label_mapping = dict(label_mapping)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.records = []
        annotation_dir = self.root / split / "ann"
        image_dir = self.root / split / "img"
        if not annotation_dir.is_dir() or not image_dir.is_dir():
            raise FileNotFoundError(f"Expected PlantDoc img/ and ann/ directories under {self.root / split}")

        unknown_targets = sorted(set(self.label_mapping.values()) - set(class_to_idx))
        if unknown_targets:
            raise ValueError(f"Mapped targets absent from checkpoint class_to_idx: {unknown_targets}")

        for annotation_path in sorted(annotation_dir.glob("*.json"), key=lambda path: path.name):
            with annotation_path.open(encoding="utf-8") as handle:
                annotation = json.load(handle)
            image_path = image_dir / annotation_path.name[:-5]
            objects = annotation.get("objects")
            if not isinstance(objects, list):
                raise ValueError(f"Missing objects list: {annotation_path}")
            for object_index, obj in enumerate(objects):
                plantdoc_class = obj.get("classTitle")
                if plantdoc_class not in self.label_mapping:
                    continue
                if obj.get("geometryType") != "rectangle":
                    raise ValueError(f"Approved object is not a rectangle: {annotation_path} object {object_index}")
                exterior = obj.get("points", {}).get("exterior", [])
                if len(exterior) != 2 or any(len(point) != 2 for point in exterior):
                    raise ValueError(f"Invalid rectangle: {annotation_path} object {object_index}")
                left, top = exterior[0]
                right, bottom = exterior[1]
                width = annotation.get("size", {}).get("width")
                height = annotation.get("size", {}).get("height")
                if not all(isinstance(value, (int, float)) for value in (left, top, right, bottom, width, height)):
                    raise ValueError(f"Non-numeric rectangle or image size: {annotation_path} object {object_index}")
                if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                    raise ValueError(f"Out-of-bounds rectangle: {annotation_path} object {object_index}")
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing image for {annotation_path}: {image_path}")
                target_class = self.label_mapping[plantdoc_class]
                self.records.append({
                    "image_path": image_path,
                    "annotation_path": annotation_path,
                    "object_index": object_index,
                    "plantdoc_class": plantdoc_class,
                    "target_class": target_class,
                    "target_index": class_to_idx[target_class],
                    "bbox": (left, top, right, bottom),
                    "annotated_size": (width, height),
                })

    @property
    def source_images(self):
        return {record["image_path"] for record in self.records}

    @property
    def plantdoc_classes(self):
        return {record["plantdoc_class"] for record in self.records}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image_path"]) as source:
            # PlantDoc annotations use the displayed orientation. Three test JPEGs
            # carry EXIF orientation 6 while their stored pixel arrays are rotated.
            source = ImageOps.exif_transpose(source)
            if source.size != record["annotated_size"]:
                raise ValueError(
                    f"Annotation/image size mismatch for {record['image_path']}: "
                    f"{record['annotated_size']} != {source.size}"
                )
            crop = source.crop(record["bbox"]).convert("RGB")
        if self.transform is not None:
            crop = self.transform(crop)
        return crop, record["target_index"], index


def validate_expected_subset(dataset, expected):
    observed = {
        "source_images": len(dataset.source_images),
        "crops": len(dataset),
        "classes": len(dataset.plantdoc_classes),
    }
    wanted = {key: int(expected[key]) for key in observed}
    if observed != wanted:
        raise ValueError(f"PlantDoc subset mismatch; refusing inference. Expected {wanted}, observed {observed}")
    return observed


def compute_metrics(targets, predictions, evaluation_indices, evaluation_names, all_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=evaluation_indices, zero_division=0
    )
    matrix = []
    for target_index in evaluation_indices:
        row = [0] * len(all_names)
        for target, prediction in zip(targets, predictions):
            if target == target_index:
                row[prediction] += 1
        matrix.append(row)
    correct = sum(target == prediction for target, prediction in zip(targets, predictions))
    per_class = []
    for position, name in enumerate(evaluation_names):
        per_class.append({
            "plantvillage_class": name,
            "class_index": evaluation_indices[position],
            "precision": float(precision[position]),
            "recall": float(recall[position]),
            "f1": float(f1[position]),
            "support": int(support[position]),
        })
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(np.mean(f1)),
        "source_images": None,
        "evaluated_crops": len(targets),
        "correct_predictions": correct,
        "incorrect_predictions": len(targets) - correct,
        "evaluation_classes": len(evaluation_indices),
        "per_class": per_class,
        "confusion_matrix": {
            "true_labels": evaluation_names,
            "predicted_labels": all_names,
            "counts": matrix,
        },
    }


def write_outputs(output, dataset, predictions, metrics, provenance):
    output.mkdir(parents=True, exist_ok=False)
    names = provenance["checkpoint"]["classes_by_index"]
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "image_path", "annotation_path", "object_index", "bbox_left", "bbox_top", "bbox_right",
            "bbox_bottom", "plantdoc_class", "plantvillage_target_class", "target_class_index",
            "predicted_plantvillage_class", "predicted_class_index", "correct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record, prediction in zip(dataset.records, predictions):
            left, top, right, bottom = record["bbox"]
            writer.writerow({
                "image_path": image_reference(record["image_path"]),
                "annotation_path": image_reference(record["annotation_path"]),
                "object_index": record["object_index"],
                "bbox_left": left, "bbox_top": top, "bbox_right": right, "bbox_bottom": bottom,
                "plantdoc_class": record["plantdoc_class"],
                "plantvillage_target_class": record["target_class"],
                "target_class_index": record["target_index"],
                "predicted_plantvillage_class": names[prediction],
                "predicted_class_index": prediction,
                "correct": prediction == record["target_index"],
            })
    matrix = metrics["confusion_matrix"]
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true / predicted", *matrix["predicted_labels"]])
        for name, row in zip(matrix["true_labels"], matrix["counts"]):
            writer.writerow([name, *row])
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def run(config_name="plantdoc_external_evaluation", device_name=None):
    config_path = project_path("configs") / f"{config_name}.yaml"
    config = load_config(config_name)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config["cpu_threads"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    checkpoint_path = project_path(config["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    preprocessing = load_config("preprocessing")["images"]
    if checkpoint["preprocessing"] != preprocessing:
        raise ValueError("Current deterministic preprocessing differs from checkpoint settings")
    class_to_idx = checkpoint["class_to_idx"]
    all_names = sorted(class_to_idx, key=class_to_idx.get)
    if list(range(len(all_names))) != [class_to_idx[name] for name in all_names]:
        raise ValueError("Checkpoint class_to_idx is not contiguous")

    plantdoc_root = resolve_plantdoc_root(config)
    dataset = PlantDocObjectDataset(
        plantdoc_root, config["split"], config["mappings"], class_to_idx, build_transforms(training=False)
    )
    observed = validate_expected_subset(dataset, config["expected"])
    print(f"Validated PlantDoc subset: {observed}", flush=True)

    device = select_device(device_name or config["device"])
    model = build_model(len(all_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.requires_grad_(False)
    model.eval()
    model.to(device)
    loader = DataLoader(
        dataset, batch_size=int(config["batch_size"]), shuffle=False,
        num_workers=int(config["num_workers"]),
    )
    targets, predictions, record_indices = [], [], []
    with torch.inference_mode():
        for images, labels, indices in progress_batches(
            loader, "PlantDoc external evaluation", int(config["log_every_batches"])
        ):
            output = model(images.to(device))
            targets.extend(labels.tolist())
            predictions.extend(output.argmax(1).cpu().tolist())
            record_indices.extend(indices.tolist())
    if record_indices != list(range(len(dataset))):
        raise RuntimeError("Evaluation loader changed deterministic record order")

    evaluation_names = list(config["mappings"].values())
    evaluation_indices = [class_to_idx[name] for name in evaluation_names]
    metrics = compute_metrics(targets, predictions, evaluation_indices, evaluation_names, all_names)
    metrics["source_images"] = len(dataset.source_images)

    source_files = [plantdoc_root / "meta.json"]
    source_files.extend((plantdoc_root / config["split"] / "ann").glob("*.json"))
    source_files.extend(dataset.source_images)
    timestamp = datetime.now(timezone.utc)
    output_root = ensure_output_path(config["output_root"])
    output = output_root / f"evaluation_{timestamp:%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    provenance = {
        "created_at_utc": timestamp.isoformat(),
        "command": f"python -m src.evaluation.evaluate_plantdoc_external --config {config_name}",
        "config": config,
        "config_sha256": sha256_file(config_path),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "preprocessing_config_sha256": sha256_file(project_path("configs/preprocessing.yaml")),
        "preprocessing": preprocessing,
        "crop_policy": (
            "Apply PIL ImageOps.exif_transpose, then crop using annotation "
            "[left, top, right, bottom] coordinates as half-open bounds"
        ),
        "checkpoint": {
            "path": checkpoint_path.relative_to(project_path(".")).as_posix(),
            "sha256": sha256_file(checkpoint_path),
            "epoch": checkpoint.get("epoch"),
            "validation_loss": checkpoint.get("validation_loss"),
            "classes_by_index": all_names,
        },
        "dataset": {
            "path": image_reference(plantdoc_root),
            "split": config["split"],
            "evaluated_source_images": len(dataset.source_images),
            "evaluated_crops": len(dataset),
            "source_collection_sha256": sha256_collection(source_files, plantdoc_root),
            "hashed_file_count": len(source_files),
        },
        "runtime": {
            "device": str(device), "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "pillow": Image.__version__, "sklearn": sklearn.__version__,
        },
        "model_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
    }
    write_outputs(output, dataset, predictions, metrics, provenance)
    print(json.dumps({key: value for key, value in metrics.items() if key != "confusion_matrix"}, indent=2))
    print(f"Results: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="plantdoc_external_evaluation")
    parser.add_argument("--device", default=None, help="Override configured device (for example, cpu or cuda)")
    args = parser.parse_args()
    run(args.config, args.device)


if __name__ == "__main__":
    main()
