"""Evaluate a frozen Week 6 candidate on PlantVillage and PlantDoc."""
import argparse
import csv
import json
import math
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import sklearn
import torch
import torchvision
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.datasets.plantvillage import create_dataset
from src.evaluation.evaluate_plantdoc_external import (
    PlantDocObjectDataset,
    resolve_plantdoc_root,
    sha256_collection,
    sha256_file,
    validate_expected_subset,
)
from src.models.resnet_baseline import build_model
from src.preprocessing.transforms import build_transforms
from src.training.train_baseline import select_device
from src.training.train_robust_augmentation import validate_fairness
from src.utils.config import (ensure_output_path, image_reference, load_config,
                              manifest_path, project_path)
from src.utils.progress import progress_batches


def repository_state():
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=project_path("."), check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def collect_outputs(model, loader, device, label, log_every):
    model.eval()
    targets, logits, record_indices = [], [], []
    total_loss = 0.0
    with torch.inference_mode():
        for batch in progress_batches(loader, label, log_every):
            images, labels, *indices = batch
            output = model(images.to(device))
            total_loss += torch.nn.functional.cross_entropy(
                output, labels.to(device), reduction="sum"
            ).item()
            targets.extend(labels.tolist())
            logits.append(output.cpu())
            if indices:
                record_indices.extend(indices[0].tolist())
    if not targets:
        raise ValueError(f"{label} loader is empty")
    return np.asarray(targets), torch.cat(logits).numpy(), record_indices, total_loss / len(targets)


def metric_block(targets, predictions, labels, names, predicted_names, loss=None):
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(predicted_names))))
    if labels != list(range(len(predicted_names))):
        matrix = matrix[labels]
    result = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(np.mean(f1)),
        "samples": int(len(targets)),
        "correct_predictions": int((targets == predictions).sum()),
        "incorrect_predictions": int((targets != predictions).sum()),
        "per_class": [
            {"class": name, "class_index": int(index), "precision": float(precision[position]),
             "recall": float(recall[position]), "f1": float(f1[position]),
             "support": int(support[position])}
            for position, (index, name) in enumerate(zip(labels, names))
        ],
        "confusion_matrix": {
            "true_labels": names, "predicted_labels": predicted_names, "counts": matrix.tolist()
        },
    }
    if loss is not None:
        result["loss"] = float(loss)
    return result


def write_metric_set(directory, metrics):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (directory / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["class", "class_index", "precision", "recall", "f1", "support"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics["per_class"])
    matrix = metrics["confusion_matrix"]
    with (directory / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true / predicted", *matrix["predicted_labels"]])
        for name, row in zip(matrix["true_labels"], matrix["counts"]):
            writer.writerow([name, *row])


def write_plantvillage_predictions(path, dataset, targets, predictions, probabilities, names):
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["image_path", "target_class", "predicted_class", "correct", "top1_confidence"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (target, prediction) in enumerate(zip(targets, predictions)):
            writer.writerow({
                "image_path": dataset.rows[index]["path"],
                "target_class": names[target], "predicted_class": names[prediction],
                "correct": target == prediction, "top1_confidence": float(probabilities[index, prediction]),
            })


def add_plantdoc_distribution_metrics(metrics, targets, predictions, probabilities, names):
    top5 = np.argsort(-probabilities, axis=1)[:, :5]
    confidence = probabilities.max(axis=1)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(axis=1)
    target_in_top5 = np.asarray([target in row for target, row in zip(targets, top5)])
    tlb_index = names.index("Tomato___Late_blight")
    metrics.update({
        "top5_accuracy": float(target_in_top5.mean()),
        "mean_top1_confidence": float(confidence.mean()),
        "median_top1_confidence": float(np.median(confidence)),
        "mean_prediction_entropy_nats": float(entropy.mean()),
        "median_prediction_entropy_nats": float(np.median(entropy)),
        "tomato_late_blight_prediction_count": int((predictions == tlb_index).sum()),
        "tomato_late_blight_prediction_frequency": float((predictions == tlb_index).mean()),
    })
    return top5, confidence, entropy


def write_plantdoc_predictions(path, dataset, targets, predictions, logits, probabilities, top5, confidence, entropy, names):
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "image_path", "annotation_path", "object_index", "bbox_left", "bbox_top", "bbox_right",
            "bbox_bottom", "plantdoc_class", "plantvillage_target_class", "target_class_index",
            "predicted_plantvillage_class", "predicted_class_index", "correct", "top1_confidence",
            "target_probability", "prediction_entropy_nats", "top5_predictions_json", "logits_json",
            "probabilities_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(dataset.records):
            left, top, right, bottom = record["bbox"]
            writer.writerow({
                "image_path": image_reference(record["image_path"]),
                "annotation_path": image_reference(record["annotation_path"]),
                "object_index": record["object_index"], "bbox_left": left, "bbox_top": top,
                "bbox_right": right, "bbox_bottom": bottom, "plantdoc_class": record["plantdoc_class"],
                "plantvillage_target_class": record["target_class"], "target_class_index": int(targets[index]),
                "predicted_plantvillage_class": names[predictions[index]],
                "predicted_class_index": int(predictions[index]), "correct": targets[index] == predictions[index],
                "top1_confidence": float(confidence[index]),
                "target_probability": float(probabilities[index, targets[index]]),
                "prediction_entropy_nats": float(entropy[index]),
                "top5_predictions_json": json.dumps([
                    {"class_index": int(class_index), "class": names[class_index],
                     "probability": float(probabilities[index, class_index])}
                    for class_index in top5[index]
                ], separators=(",", ":")),
                "logits_json": json.dumps(logits[index].tolist(), separators=(",", ":")),
                "probabilities_json": json.dumps(probabilities[index].tolist(), separators=(",", ":")),
            })


def baseline_top5(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        row = next(row for row in csv.DictReader(handle) if row["variant"] == "TIGHT")
    return float(row["top5_accuracy"])


def build_comparison(config, plantvillage, plantdoc):
    baseline = config["baseline"]
    baseline_pv = json.loads(project_path(baseline["plantvillage_metrics"]).read_text(encoding="utf-8"))
    baseline_pd = json.loads(project_path(baseline["plantdoc_metrics"]).read_text(encoding="utf-8"))
    base_top5 = baseline_top5(project_path(baseline["confidence_summary"]))
    rows = [
        ("PlantVillage", "accuracy", baseline_pv["accuracy"], plantvillage["accuracy"]),
        ("PlantVillage", "macro_f1", baseline_pv["macro_f1"], plantvillage["macro_f1"]),
        ("PlantDoc", "accuracy", baseline_pd["accuracy"], plantdoc["accuracy"]),
        ("PlantDoc", "macro_f1", baseline_pd["macro_f1"], plantdoc["macro_f1"]),
        ("PlantDoc", "top5_accuracy", base_top5, plantdoc["top5_accuracy"]),
    ]
    return [{
        "dataset": dataset, "metric": metric, "baseline": float(old), "robust": float(new),
        "change": float(new - old),
        "change_percentage_points": float(100 * (new - old)) if "accuracy" in metric else None,
    } for dataset, metric, old, new in rows]


def write_comparison(output, rows):
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lookup = {(row["dataset"], row["metric"]): row for row in rows}
    pv_a, pv_f = lookup[("PlantVillage", "accuracy")], lookup[("PlantVillage", "macro_f1")]
    pd_a, pd_f, pd_5 = (lookup[("PlantDoc", "accuracy")], lookup[("PlantDoc", "macro_f1")],
                         lookup[("PlantDoc", "top5_accuracy")])
    report = f"""# Week 6 robust-augmentation comparison

This report records the measured trade-off without assigning a success/failure label.

| Dataset | Metric | Week 5 baseline | Robust augmentation | Change |
|---|---:|---:|---:|---:|
| PlantVillage | Accuracy | {pv_a['baseline']:.6f} | {pv_a['robust']:.6f} | {pv_a['change_percentage_points']:+.4f} pp |
| PlantVillage | Macro F1 | {pv_f['baseline']:.6f} | {pv_f['robust']:.6f} | {pv_f['change']:+.6f} |
| PlantDoc | Accuracy | {pd_a['baseline']:.6f} | {pd_a['robust']:.6f} | {pd_a['change_percentage_points']:+.4f} pp |
| PlantDoc | Macro F1 | {pd_f['baseline']:.6f} | {pd_f['robust']:.6f} | {pd_f['change']:+.6f} |
| PlantDoc | Top-5 accuracy | {pd_5['baseline']:.6f} | {pd_5['robust']:.6f} | {pd_5['change_percentage_points']:+.4f} pp |

PlantDoc remained an unseen evaluation domain: its images were used only after training.
"""
    (output / "analysis.md").write_text(report, encoding="utf-8")


def run(checkpoint_name, config_name="robust_augmentation_evaluation", device_name=None):
    config_path = project_path("configs") / f"{config_name}.yaml"
    config = load_config(config_name)
    checkpoint_path = project_path(checkpoint_name).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("smoke"):
        raise ValueError("Smoke checkpoints are pipeline checks and cannot produce measured Week 6 results")
    validate_fairness(checkpoint["config"])
    if checkpoint["config"].get("augmentation", {}).get("profile") != "field_style_v1":
        raise ValueError("Checkpoint is not the Week 6 field-style candidate")
    preprocessing = load_config("preprocessing")["images"]
    if checkpoint["preprocessing"] != preprocessing:
        raise ValueError("Deterministic preprocessing differs from checkpoint metadata")
    for split, manifest in checkpoint["config"]["manifests"].items():
        if sha256_file(manifest_path(manifest)) != checkpoint["manifest_sha256"][split]:
            raise ValueError(f"Manifest changed since training: {manifest}")

    seed = int(config["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(int(config["cpu_threads"]))
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    device = select_device(device_name or config["device"])
    class_to_idx = checkpoint["class_to_idx"]
    names = sorted(class_to_idx, key=class_to_idx.get)
    model = build_model(len(names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.requires_grad_(False); model.eval(); model.to(device)

    pv_dataset = create_dataset("test", build_transforms(False), class_to_idx)
    pv_loader = DataLoader(pv_dataset, batch_size=int(config["batch_size"]), shuffle=False,
                           num_workers=int(config["num_workers"]))
    pv_targets, pv_logits, _, pv_loss = collect_outputs(
        model, pv_loader, device, "PlantVillage test", int(config["log_every_batches"])
    )
    pv_probabilities = torch.softmax(torch.from_numpy(pv_logits), dim=1).numpy()
    pv_predictions = pv_logits.argmax(axis=1)
    pv_metrics = metric_block(pv_targets, pv_predictions, list(range(len(names))), names, names, pv_loss)

    plantdoc_config = load_config(config["plantdoc_config"])
    plantdoc_root = resolve_plantdoc_root(plantdoc_config)
    pd_dataset = PlantDocObjectDataset(
        plantdoc_root, plantdoc_config["split"], plantdoc_config["mappings"],
        class_to_idx, build_transforms(False),
    )
    observed = validate_expected_subset(pd_dataset, plantdoc_config["expected"])
    pd_loader = DataLoader(pd_dataset, batch_size=int(config["batch_size"]), shuffle=False,
                           num_workers=int(config["num_workers"]))
    pd_targets, pd_logits, record_indices, pd_loss = collect_outputs(
        model, pd_loader, device, "PlantDoc external test", int(config["log_every_batches"])
    )
    if record_indices != list(range(len(pd_dataset))):
        raise RuntimeError("PlantDoc loader changed deterministic record order")
    pd_probabilities = torch.softmax(torch.from_numpy(pd_logits), dim=1).numpy()
    pd_predictions = pd_logits.argmax(axis=1)
    evaluation_names = list(plantdoc_config["mappings"].values())
    evaluation_indices = [class_to_idx[name] for name in evaluation_names]
    pd_metrics = metric_block(
        pd_targets, pd_predictions, evaluation_indices, evaluation_names, names, pd_loss
    )
    pd_metrics.update({"source_images": observed["source_images"], "evaluated_crops": observed["crops"],
                       "evaluation_classes": observed["classes"]})
    top5, confidence, entropy = add_plantdoc_distribution_metrics(
        pd_metrics, pd_targets, pd_predictions, pd_probabilities, names
    )

    timestamp = datetime.now(timezone.utc)
    output = ensure_output_path(config["output_root"]) / f"evaluation_{timestamp:%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    write_metric_set(output / "plantvillage", pv_metrics)
    write_plantvillage_predictions(
        output / "plantvillage" / "predictions.csv", pv_dataset, pv_targets, pv_predictions,
        pv_probabilities, names,
    )
    write_metric_set(output / "plantdoc", pd_metrics)
    write_plantdoc_predictions(
        output / "plantdoc" / "predictions.csv", pd_dataset, pd_targets, pd_predictions,
        pd_logits, pd_probabilities, top5, confidence, entropy, names,
    )
    comparison = build_comparison(config, pv_metrics, pd_metrics)
    write_comparison(output, comparison)

    plantdoc_sources = [plantdoc_root / "meta.json"]
    plantdoc_sources.extend((plantdoc_root / plantdoc_config["split"] / "ann").glob("*.json"))
    plantdoc_sources.extend(pd_dataset.source_images)
    training_history = checkpoint_path.parent / "history.json"
    provenance = {
        "created_at_utc": timestamp.isoformat(),
        "command": f"python -B -m src.evaluation.evaluate_robust_augmentation --checkpoint {checkpoint_name}",
        "repository": repository_state(),
        "config": config, "config_sha256": sha256_file(config_path),
        "training_config_sha256": sha256_file(project_path("configs/robust_augmentation.yaml")),
        "checkpoint": {"path": checkpoint_name, "sha256": sha256_file(checkpoint_path),
                       "epoch": checkpoint.get("epoch"), "validation_loss": checkpoint.get("validation_loss")},
        "training_history_sha256": sha256_file(training_history) if training_history.is_file() else None,
        "baseline_checkpoint_sha256": sha256_file(project_path(config["baseline"]["checkpoint"])),
        "baseline_reference_sha256": {
            key: sha256_file(project_path(path)) for key, path in config["baseline"].items()
        },
        "manifest_sha256": checkpoint["manifest_sha256"],
        "plantdoc_source_collection_sha256": sha256_collection(plantdoc_sources, plantdoc_root),
        "plantdoc_hashed_file_count": len(plantdoc_sources),
        "preprocessing": preprocessing, "training_augmentation": checkpoint["config"]["augmentation"],
        "model_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "plantdoc_used_for_training": False,
        "runtime": {"device": str(device), "python": platform.python_version(), "torch": torch.__version__,
                    "torchvision": torchvision.__version__, "pillow": Image.__version__,
                    "sklearn": sklearn.__version__},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "plantvillage": {"accuracy": pv_metrics["accuracy"],
          "macro_f1": pv_metrics["macro_f1"]}, "plantdoc": {key: pd_metrics[key] for key in
          ("accuracy", "macro_f1", "top5_accuracy", "mean_top1_confidence",
           "mean_prediction_entropy_nats", "tomato_late_blight_prediction_frequency")}}, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="robust_augmentation_evaluation")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(args.checkpoint, args.config, args.device)


if __name__ == "__main__":
    main()
