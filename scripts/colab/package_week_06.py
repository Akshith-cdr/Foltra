"""Package Week 6 datasets and immutable baseline references for Google Drive."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from src.utils.config import PROJECT_ROOT, dataset_path


def add_tree(archive, source, archive_root):
    for count, path in enumerate(sorted(item for item in source.rglob("*") if item.is_file()), 1):
        archive.write(path, (archive_root / path.relative_to(source)).as_posix())
        if count % 5000 == 0:
            print(f"Packed {count} files from {source.name}", flush=True)


def main():
    destination = PROJECT_ROOT / "experiments/week_06_robust_augmentation/training/outputs/colab_upload"
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "plantvillage_color.zip": destination / "plantvillage_color.zip",
        "plantdoc_test.zip": destination / "plantdoc_test.zip",
        "baseline_reference.zip": destination / "baseline_reference.zip",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing archives: {existing}")

    plantvillage = dataset_path("plantvillage")
    if not plantvillage.is_dir():
        raise FileNotFoundError(plantvillage)
    with ZipFile(outputs["plantvillage_color.zip"], "x", ZIP_STORED) as archive:
        add_tree(archive, plantvillage, Path("Datasets/plantvillage/color"))

    plantdoc = dataset_path("plantdoc")
    required = [plantdoc / "meta.json", plantdoc / "test" / "img", plantdoc / "test" / "ann"]
    if not all(path.exists() for path in required):
        raise FileNotFoundError(f"Incomplete PlantDoc test inputs: {required}")
    with ZipFile(outputs["plantdoc_test.zip"], "x", ZIP_STORED) as archive:
        archive.write(plantdoc / "meta.json", "Datasets/plantdoc/meta.json")
        add_tree(archive, plantdoc / "test" / "img", Path("Datasets/plantdoc/test/img"))
        add_tree(archive, plantdoc / "test" / "ann", Path("Datasets/plantdoc/test/ann"))

    references = [
        "experiments/week_05_baseline/outputs/train_20260916T171709Z_f582a465/best.pt",
        "experiments/week_05_baseline/outputs/train_20260916T171709Z_f582a465/test/metrics.json",
        "experiments/plantdoc_external_evaluation/outputs/evaluation_20260924T061804Z_3d272ea1/metrics.json",
        "experiments/plantdoc_external_evaluation/confidence_analysis/confidence_summary.csv",
    ]
    with ZipFile(outputs["baseline_reference.zip"], "x", ZIP_DEFLATED) as archive:
        for relative in references:
            path = PROJECT_ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.write(path, relative)

    for name, path in outputs.items():
        print(f"{name}: {path}", flush=True)


if __name__ == "__main__":
    main()

