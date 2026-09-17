"""Classification metrics and standalone checkpoint evaluation."""
import argparse
import csv
import hashlib
import json

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from src.utils.progress import progress_batches


def evaluate(model, loader, device, num_classes, progress_label=None, log_every_batches=20):
    model.eval()
    total_loss = 0.0
    targets, predictions = [], []
    with torch.inference_mode():
        batches = progress_batches(loader, progress_label, log_every_batches) if progress_label else loader
        for images, labels in batches:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    if not targets:
        raise ValueError("Evaluation loader is empty")
    classes = list(range(num_classes))
    return {
        "loss": total_loss / len(targets),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=classes, average="macro", zero_division=0)),
        "samples": len(targets),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=classes).tolist(),
    }


def save_metrics(metrics, class_to_idx, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    names = sorted(class_to_idx, key=class_to_idx.get)
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true / predicted", *names])
        for name, row in zip(names, metrics["confusion_matrix"]):
            writer.writerow([name, *row])


def main():
    from src.models.resnet_baseline import build_model
    from src.training.train_baseline import build_datasets, make_loader, select_device, new_output
    from src.utils.config import project_path, load_config, manifest_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    checkpoint = torch.load(project_path(args.checkpoint), map_location="cpu", weights_only=True)
    if checkpoint["preprocessing"] != load_config("preprocessing")["images"]:
        raise ValueError("Current preprocessing differs from the checkpoint settings")
    config = checkpoint["config"]
    for split, name in config["manifests"].items():
        if hashlib.sha256(manifest_path(name).read_bytes()).hexdigest() != checkpoint["manifest_sha256"][split]:
            raise ValueError(f"Manifest changed since training: {name}")
    torch.set_num_threads(config["cpu_threads"])
    datasets, mapping = build_datasets(config, checkpoint["smoke"], checkpoint["class_to_idx"])
    device = select_device(args.device)
    model = build_model(len(mapping), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate(model, make_loader(datasets["test"], config, False), device, len(mapping), "Test")
    output = new_output(config, "evaluation")
    save_metrics(metrics, mapping, output / "test")
    print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}, indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
