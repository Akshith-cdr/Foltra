"""Explicit image verification for class-folder datasets. Never changes raw files."""
import argparse
from collections import Counter
from pathlib import Path
from PIL import Image
from src.utils.config import dataset_path, load_config


def inspect_dataset(root, limit=None, verify=False):
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    extensions = set(load_config("preprocessing")["images"]["extensions"])
    sizes, modes, classes = Counter(), Counter(), Counter()
    failures = []; total = 0
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            if limit is not None and total >= limit:
                return {"total": total, "classes": dict(classes), "resolutions": dict(sizes), "modes": dict(modes), "failures": failures}
            total += 1
            try:
                with Image.open(path) as image:
                    size, mode = image.size, image.mode
                    if verify:
                        image.verify()
                sizes[str(size)] += 1; modes[mode] += 1; classes[folder.name] += 1
            except Exception as error:
                failures.append({"path": str(path), "error": str(error)})
    return {"total": total, "classes": dict(classes), "resolutions": dict(sizes), "modes": dict(modes), "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["cassava", "plantvillage"])
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--limit", type=int)
    scope.add_argument("--all", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(inspect_dataset(dataset_path(args.dataset), args.limit, args.verify))


if __name__ == "__main__":
    main()
