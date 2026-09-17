"""Create upload archives without changing datasets or manifests."""
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED, ZIP_DEFLATED
from src.utils.config import PROJECT_ROOT, dataset_path


def main():
    destination = PROJECT_ROOT / "experiments/week_05_baseline/outputs/colab_upload"
    destination.mkdir(parents=True, exist_ok=True)
    code = destination / "foltra_code.zip"
    data = destination / "plantvillage_color.zip"
    if code.exists() or data.exists():
        raise FileExistsError("Upload archives already exist; rename them before packaging again")
    with ZipFile(code, "x", ZIP_DEFLATED) as archive:
        for folder in ("src", "configs", "data/manifests"):
            for path in sorted((PROJECT_ROOT / folder).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(PROJECT_ROOT).as_posix())
        archive.write(PROJECT_ROOT / "requirements.txt", "requirements.txt")
    print(f"Code archive: {code}", flush=True)
    color = dataset_path("plantvillage")
    if not color.is_dir():
        raise FileNotFoundError(color)
    with ZipFile(data, "x", ZIP_STORED) as archive:
        for count, path in enumerate(sorted(p for p in color.rglob("*") if p.is_file()), 1):
            archive.write(path, (Path("Datasets/plantvillage/color") / path.relative_to(color)).as_posix())
            if count % 5000 == 0:
                print(f"Packed {count} files", flush=True)
    print(f"Dataset archive: {data}", flush=True)


if __name__ == "__main__":
    main()
