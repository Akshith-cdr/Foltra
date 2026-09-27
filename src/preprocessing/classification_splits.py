"""Deterministic NEW classification splits. Historical memberships are never regenerated here."""
import argparse
import csv
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.utils.config import dataset_path, image_reference, load_config, ensure_output_path


def collect_samples(root):
    extensions = set(load_config("preprocessing")["images"]["extensions"])
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    return [{"path": image_reference(p), "label": folder.name}
            for folder in sorted(root.iterdir()) if folder.is_dir()
            for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in extensions]


def split_samples(samples, seed=42, ratios=(0.7, 0.15, 0.15)):
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios)-1) > 1e-8:
        raise ValueError("Three positive split ratios must sum to one")
    if not samples:
        raise ValueError("No classification samples")
    labels = [s["label"] for s in samples]
    train, other = train_test_split(samples, test_size=ratios[1]+ratios[2],
                                    stratify=labels, random_state=seed)
    val, test = train_test_split(other, test_size=ratios[2]/(ratios[1]+ratios[2]),
                                stratify=[s["label"] for s in other], random_state=seed)
    return {"train": train, "val": val, "test": test}


def create_split(dataset, output_dir):
    cfg = load_config("preprocessing")
    options = cfg["classification"]
    if dataset not in options["supported_datasets"]:
        raise ValueError("Only Cassava and PlantVillage color splitting is implemented")
    if not options["stratify"]:
        raise ValueError("This implementation requires classification stratification")
    prefix = load_config()["datasets"][dataset]["manifest_prefix"]
    destination = ensure_output_path(output_dir)
    targets = {s: destination / f"{prefix}_{s}.csv" for s in ("train", "val", "test")}
    if any(p.exists() for p in targets.values()):
        raise FileExistsError("Refusing to overwrite existing manifests; choose a fresh output directory")
    if dataset == "plantvillage" and load_config()["datasets"][dataset]["representation"] != "color":
        raise ValueError("Only the color representation is validated for this splitter")
    rows = collect_samples(dataset_path(dataset))
    splits = split_samples(rows, cfg["random_seed"], tuple(options[s+"_ratio"] for s in ("train", "val", "test")))
    destination.mkdir(parents=True, exist_ok=True)
    for split, samples in splits.items():
        with targets[split].open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "label"])
            writer.writeheader(); writer.writerows(samples)
    return {s: len(rows) for s, rows in splits.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["cassava", "plantvillage"])
    parser.add_argument("--output-dir", required=True, help="Fresh repository-relative output directory")
    args = parser.parse_args()
    print(create_split(args.dataset, args.output_dir))


if __name__ == "__main__":
    main()
