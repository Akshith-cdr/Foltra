"""Create the data-only Google Colab bundle for the Week 06 experiment.

Application code is deliberately not packaged. Colab clones one exact Git commit,
which prevents a stale ZIP from silently replacing newer evaluator code.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from src.utils.config import PROJECT_ROOT, dataset_path


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments/week_06_robust_augmentation/training/outputs/colab_bundle_v2"
)

REQUIRED_TRACKED_FILES = {
    "configs/datasets.yaml",
    "configs/plantdoc_external_evaluation.yaml",
    "configs/preprocessing.yaml",
    "configs/robust_augmentation.yaml",
    "configs/robust_augmentation_evaluation.yaml",
    "data/manifests/plantvillage_train.csv",
    "data/manifests/plantvillage_val.csv",
    "data/manifests/plantvillage_test.csv",
    "notebooks/week_06_robust_augmentation_colab.ipynb",
    "requirements.txt",
    "src/evaluation/evaluate_plantdoc_external.py",
    "src/evaluation/evaluate_robust_augmentation.py",
    "src/preprocessing/transforms.py",
    "src/training/train_baseline.py",
    "src/training/train_robust_augmentation.py",
    "src/utils/config.py",
}

BASELINE_REFERENCES = {
    "experiments/week_05_baseline/outputs/train_20260916T171709Z_f582a465/best.pt",
    "experiments/week_05_baseline/outputs/train_20260916T171709Z_f582a465/test/metrics.json",
    "experiments/plantdoc_external_evaluation/outputs/evaluation_20260924T061804Z_3d272ea1/metrics.json",
    "experiments/plantdoc_external_evaluation/confidence_analysis/confidence_summary.csv",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_files():
    result = subprocess.run(
        ["git", "ls-files"], cwd=PROJECT_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return set(result.stdout.splitlines())


def changed_files():
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return {line[3:].strip().replace("\\", "/") for line in result.stdout.splitlines()}


def verify_sources_tracked(files=None, changed=None):
    observed = tracked_files() if files is None else set(files)
    missing = sorted(REQUIRED_TRACKED_FILES - observed)
    if missing:
        joined = "\n  - ".join(missing)
        raise RuntimeError(
            "Week 06 source files must be committed before packaging. Missing from Git:\n"
            f"  - {joined}\nCommit these files, then rerun the packager."
        )
    dirty = changed_files() if changed is None else set(changed)
    uncommitted = sorted(REQUIRED_TRACKED_FILES & dirty)
    if uncommitted:
        joined = "\n  - ".join(uncommitted)
        raise RuntimeError(
            "Week 06 source files must match the Git commit used by Colab. "
            "Uncommitted changes:\n"
            f"  - {joined}\nCommit these changes, then rerun the packager."
        )
    return sorted(REQUIRED_TRACKED_FILES)


def add_tree(archive, source, archive_root):
    files = sorted(path for path in source.rglob("*") if path.is_file())
    for count, path in enumerate(files, 1):
        archive.write(path, (archive_root / path.relative_to(source)).as_posix())
        if count % 5000 == 0:
            print(f"Packed {count}/{len(files)} files from {source.name}", flush=True)
    return len(files)


def build_bundle(destination):
    verify_sources_tracked()
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(
            f"Bundle directory is not empty: {destination}. Use a new --output-dir."
        )

    plantvillage = dataset_path("plantvillage")
    plantdoc = dataset_path("plantdoc")
    required_data = [
        plantvillage,
        plantdoc / "meta.json",
        plantdoc / "test" / "img",
        plantdoc / "test" / "ann",
    ]
    missing_data = [str(path) for path in required_data if not path.exists()]
    if missing_data:
        raise FileNotFoundError(f"Missing dataset inputs: {missing_data}")

    reference_paths = {name: PROJECT_ROOT / name for name in BASELINE_REFERENCES}
    missing_references = [name for name, path in reference_paths.items() if not path.is_file()]
    if missing_references:
        raise FileNotFoundError(f"Missing baseline reference artifacts: {missing_references}")

    outputs = {
        "plantvillage_color.zip": destination / "plantvillage_color.zip",
        "plantdoc_test.zip": destination / "plantdoc_test.zip",
        "baseline_reference.zip": destination / "baseline_reference.zip",
    }
    counts = {}
    with ZipFile(outputs["plantvillage_color.zip"], "x", ZIP_STORED) as archive:
        counts["plantvillage_files"] = add_tree(
            archive, plantvillage, Path("Datasets/plantvillage/color")
        )
    with ZipFile(outputs["plantdoc_test.zip"], "x", ZIP_STORED) as archive:
        archive.write(plantdoc / "meta.json", "Datasets/plantdoc/meta.json")
        counts["plantdoc_images"] = add_tree(
            archive, plantdoc / "test" / "img", Path("Datasets/plantdoc/test/img")
        )
        counts["plantdoc_annotations"] = add_tree(
            archive, plantdoc / "test" / "ann", Path("Datasets/plantdoc/test/ann")
        )
    with ZipFile(outputs["baseline_reference.zip"], "x", ZIP_DEFLATED) as archive:
        for relative, path in sorted(reference_paths.items()):
            archive.write(path, relative)

    manifest = {
        "format": "foltra_week_06_colab_bundle_v2",
        "source_policy": "Code must come only from the exact Git commit selected in Colab.",
        "required_tracked_files": sorted(REQUIRED_TRACKED_FILES),
        "counts": counts,
        "archives": {
            name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for name, path in outputs.items()
        },
    }
    manifest_path = destination / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Bundle created: {destination}", flush=True)
    for name, path in outputs.items():
        print(f"  {name}: {path}", flush=True)
    print(f"  bundle_manifest.json: {manifest_path}", flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    build_bundle(args.output_dir)


if __name__ == "__main__":
    main()
