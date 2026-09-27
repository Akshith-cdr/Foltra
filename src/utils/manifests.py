"""Read historical manifests without changing row order or sample membership."""
import csv
from .config import manifest_path, portable_path, resolve_image_path


def load_manifest(path, required=()):
    from pathlib import Path
    path = Path(path)
    if not path.is_absolute():
        path = manifest_path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed CSV row in {path}")
        for key, value in row.items():
            if key == "path" or key.endswith("_path"):
                if not value.strip():
                    raise ValueError(f"Blank {key} in {path}")
                row[key] = portable_path(value)
    return rows


def validate_manifest(path, required=(), check_files=False):
    rows = load_manifest(path, required)
    references = [value for row in rows for key, value in row.items()
                  if key == "path" or key.endswith("_path")]
    resolved = [resolve_image_path(value) for value in references]
    missing = [str(path) for path in resolved if check_files and not path.is_file()]
    return {"rows": len(rows), "references": len(references), "missing": missing}
