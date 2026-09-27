"""Check configured paths without scanning or decoding datasets."""
from src.utils.config import configured_path, load_config


def main():
    load_config("preprocessing")
    for key in ("dataset_root", "manifest_root", "processed_root"):
        path = configured_path(key)
        print(f"{key}: {path} (exists={path.exists()})")
    if not configured_path("dataset_root").is_dir():
        print("Raw dataset setup is pending. Paste dataset folders into Foltra/Datasets, or configure ../Datasets.")


if __name__ == "__main__":
    main()
