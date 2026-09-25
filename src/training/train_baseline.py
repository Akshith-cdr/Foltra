"""Train the Week 5 PlantVillage Color baseline using preserved manifests."""
import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.datasets.classification import ClassificationDataset
from src.datasets.plantvillage import create_dataset
from src.evaluation.evaluate import evaluate, save_metrics
from src.models.resnet_baseline import build_model
from src.preprocessing.transforms import build_transforms
from src.utils.config import load_config, manifest_path, project_path, ensure_output_path
from src.utils.progress import progress_batches


def select_device(name):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def build_datasets(config, smoke=False, class_to_idx=None):
    datasets = {}
    mapping = class_to_idx
    for split in ("train", "val", "test"):
        transform = build_transforms(
            training=split == "train",
            augmentation=config.get("augmentation") if split == "train" else None,
        )
        manifest = config["manifests"][split]
        if manifest == f"plantvillage_{split}.csv":
            dataset = create_dataset(split, transform, mapping)
        else:
            dataset = ClassificationDataset(manifest, transform, mapping)
        if split == "train":
            mapping = dataset.class_to_idx
        if smoke:
            limit = config["smoke"]["samples_per_split"]
            if limit < 1:
                raise ValueError("Smoke sample limit must be positive")
            indices = random.Random(config["seed"]).sample(range(len(dataset)), min(limit, len(dataset)))
            dataset = Subset(dataset, indices)
        datasets[split] = dataset
    return datasets, mapping


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, config, training):
    cfg = config["training"]
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=training,
                      num_workers=cfg["num_workers"], worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(config["seed"]))


def new_output(config, mode):
    root = ensure_output_path(config["output_root"])
    experiments = project_path("experiments").resolve()
    if experiments not in root.parents or "outputs" not in root.relative_to(experiments).parts:
        raise ValueError("output_root must be under experiments/<experiment>/outputs")
    output = root / f"{mode}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def run(config, smoke=False):
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cfg = dict(config["training"])
    if smoke:
        cfg.update({key: config["smoke"][key] for key in ("epochs", "batch_size")})
    if cfg["epochs"] < 1 or cfg["batch_size"] < 1 or cfg["learning_rate"] <= 0:
        raise ValueError("Epochs, batch size and learning rate must be positive")
    runtime_config = {**config, "training": cfg}
    datasets, mapping = build_datasets(runtime_config, smoke)
    loaders = {split: make_loader(dataset, runtime_config, split == "train") for split, dataset in datasets.items()}
    device = select_device(config["device"])
    print(f"Device: {device}; CPU threads: {config['cpu_threads']}; "
          f"loader workers: {cfg['num_workers']}; samples: "
          f"{ {split: len(dataset) for split, dataset in datasets.items()} }", flush=True)
    output = new_output(config, "smoke" if smoke else "train")
    # Keep pretrained downloads in the ignored experiment output tree.
    torch.hub.set_dir(str(ensure_output_path(config["output_root"]) / "torch_cache"))
    metadata = {
        "config": config, "effective_training": cfg, "smoke": smoke, "class_to_idx": mapping,
        "preprocessing": load_config("preprocessing")["images"],
        "manifest_sha256": {split: hashlib.sha256(manifest_path(name).read_bytes()).hexdigest()
                            for split, name in config["manifests"].items()},
        "samples": {split: len(dataset) for split, dataset in datasets.items()},
        "device": str(device), "torch_version": str(torch.__version__),
        "weights": "ResNet18_Weights.IMAGENET1K_V1", "status": "running",
        "training_augmentation": config.get("augmentation", {"profile": "week_05_baseline"}),
    }
    record = output / "run.json"
    record.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        print("Loading pretrained ResNet-18...", flush=True)
        model = build_model(len(mapping)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        best_loss = float("inf")
        history = []
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            total, count = 0.0, 0
            for images, labels in progress_batches(loaders["train"], f"Epoch {epoch}/{cfg['epochs']} train",
                                                   cfg.get("log_every_batches", 20)):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(images), labels)
                loss.backward()
                optimizer.step()
                total += loss.item() * labels.size(0)
                count += labels.size(0)
            validation = evaluate(model, loaders["val"], device, len(mapping), "Validation",
                                  cfg.get("log_every_batches", 20))
            row = {"epoch": epoch, "train_loss": total / count, "validation_loss": validation["loss"],
                   "validation_accuracy": validation["accuracy"]}
            history.append(row)
            print(json.dumps(row), flush=True)
            (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if validation["loss"] < best_loss:
                best_loss = validation["loss"]
                torch.save({**metadata, "epoch": epoch, "validation_loss": best_loss,
                            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
                           output / "best.pt")
        checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics = evaluate(model, loaders["test"], device, len(mapping), "Test",
                           cfg.get("log_every_batches", 20))
        save_metrics(metrics, mapping, output / "test")
        metadata.update(status="completed", best_epoch=checkpoint["epoch"])
        print(json.dumps({"test_accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}), flush=True)
        print(f"Results: {output}", flush=True)
    except KeyboardInterrupt:
        metadata.update(status="interrupted")
        print(f"Training interrupted. Existing artifacts: {output}", flush=True)
        raise
    except Exception as exc:
        metadata.update(status="failed", error=str(exc))
        raise
    finally:
        record.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="baseline", help="Configuration name under configs/, without .yaml")
    parser.add_argument("--smoke", action="store_true", help="Use bounded samples and smoke hyperparameters")
    args = parser.parse_args()
    run(load_config(args.config), args.smoke)


if __name__ == "__main__":
    main()
