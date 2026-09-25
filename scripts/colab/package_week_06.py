"""Package Week 6 code, datasets, and immutable baseline references for Colab."""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from src.utils.config import PROJECT_ROOT, dataset_path


CODE_DIRECTORIES = (
    "src", "scripts", "configs", "tests", "data/manifests",
    "experiments/week_06_robust_augmentation/configuration",
    "experiments/week_06_robust_augmentation/analysis",
)
CODE_FILES = (
    "requirements.txt", "pyproject.toml", "README.md",
    "notebooks/week_06_robust_augmentation_colab.ipynb",
)
REQUIRED_CODE_MEMBERS = {
    "src/evaluation/evaluate_plantdoc_external.py",
    "src/evaluation/evaluate_robust_augmentation.py",
    "src/training/train_baseline.py",
    "src/training/train_robust_augmentation.py",
    "src/preprocessing/transforms.py",
    "src/models/resnet_baseline.py",
    "src/datasets/classification.py",
    "src/datasets/plantvillage.py",
    "src/utils/config.py",
    "configs/robust_augmentation.yaml",
    "configs/robust_augmentation_evaluation.yaml",
    "configs/plantdoc_external_evaluation.yaml",
    "configs/preprocessing.yaml",
    "configs/datasets.yaml",
    "data/manifests/plantvillage_train.csv",
    "data/manifests/plantvillage_val.csv",
    "data/manifests/plantvillage_test.csv",
    "tests/test_core.py",
    "tests/test_plantdoc_external.py",
    "tests/test_robust_augmentation.py",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_files():
    """Return the existing project files needed by the Week 6 Colab workflow."""
    files = []
    for relative in CODE_DIRECTORIES:
        directory = PROJECT_ROOT / relative
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        files.extend(
            path for path in directory.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
        )
    for relative in CODE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(path)
    return sorted(set(files), key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def write_code_archive(path):
    files = code_files()
    members = {item.relative_to(PROJECT_ROOT).as_posix(): sha256(item) for item in files}
    missing = sorted(REQUIRED_CODE_MEMBERS - set(members))
    if missing:
        raise FileNotFoundError(f"Week 6 code package is missing required members: {missing}")
    manifest = {
        "format": "foltra_week_06_code_v1",
        "files": members,
        "required_members_verified": sorted(REQUIRED_CODE_MEMBERS),
    }
    with ZipFile(path, "x", ZIP_DEFLATED) as archive:
        for item in files:
            archive.write(item, item.relative_to(PROJECT_ROOT).as_posix())
        archive.writestr("week_06_package_manifest.json", json.dumps(manifest, indent=2))
    return manifest


def add_tree(archive, source, archive_root):
    for count, path in enumerate(sorted(item for item in source.rglob("*") if item.is_file()), 1):
        archive.write(path, (archive_root / path.relative_to(source)).as_posix())
        if count % 5000 == 0:
            print(f"Packed {count} files from {source.name}", flush=True)


def main():
    destination = PROJECT_ROOT / "experiments/week_06_robust_augmentation/training/outputs/colab_upload"
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        "foltra_week06_code.zip": destination / "foltra_week06_code.zip",
        "plantvillage_color.zip": destination / "plantvillage_color.zip",
        "plantdoc_test.zip": destination / "plantdoc_test.zip",
        "baseline_reference.zip": destination / "baseline_reference.zip",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing archives: {existing}")

    reference_names = [
        "experiments/week_05_baseline/outputs/train_20260916T171709Z_f582a465/best.pt",
        "experiments/week_05_baseline/outputs/train_20260916T171709Z_f582a465/test/metrics.json",
        "experiments/plantdoc_external_evaluation/outputs/evaluation_20260924T061804Z_3d272ea1/metrics.json",
        "experiments/plantdoc_external_evaluation/confidence_analysis/confidence_summary.csv",
    ]
    references = [PROJECT_ROOT / relative for relative in reference_names]
    missing_references = [str(path) for path in references if not path.is_file()]
    if missing_references:
        raise FileNotFoundError(f"Missing baseline references: {missing_references}")

    manifest = write_code_archive(outputs["foltra_week06_code.zip"])
    print(f"Verified {len(manifest['files'])} code-package files", flush=True)

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

    with ZipFile(outputs["baseline_reference.zip"], "x", ZIP_DEFLATED) as archive:
        for relative, path in zip(reference_names, references):
            archive.write(path, relative)

    for name, path in outputs.items():
        print(f"{name}: {path}", flush=True)


if __name__ == "__main__":
    main()
