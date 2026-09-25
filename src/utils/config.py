"""Resolve configured paths from the repository, never the working directory."""
import os
from pathlib import Path, PureWindowsPath

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(name="datasets"):
    path = PROJECT_ROOT / "configs" / f"{name}.yaml"
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def configured_path(key):
    if key == "dataset_root" and os.environ.get("FOLTRA_DATA_ROOT"):
        return Path(os.environ["FOLTRA_DATA_ROOT"]).expanduser().resolve()
    return project_path(load_config()["paths"][key])


def dataset_path(name):
    config = load_config()
    return configured_path("dataset_root") / config["datasets"][name]["path"]


def manifest_path(name):
    return configured_path("manifest_root") / name


def portable_path(value):
    """Normalize historical Windows relative paths; reject machine-specific paths."""
    value = str(value).replace("\\", "/")
    if value.startswith("/") or Path(value).is_absolute() or PureWindowsPath(value).drive or ".." in Path(value).parts:
        raise ValueError(f"Expected a portable relative path: {value}")
    return value


def resolve_image_path(value):
    """Datasets/... is a logical prefix, mapped to the configured dataset root."""
    value = portable_path(value)
    parts = Path(value).parts
    if not parts or parts[0] != "Datasets":
        raise ValueError(f"Expected a Datasets/ manifest path: {value}")
    return configured_path("dataset_root").joinpath(*parts[1:])


def image_reference(path):
    relative = Path(path).resolve().relative_to(configured_path("dataset_root").resolve())
    return (Path("Datasets") / relative).as_posix()


def ensure_output_path(path):
    """Prevent report/manifest writers from touching the raw dataset tree."""
    path = project_path(path).resolve()
    raw = configured_path("dataset_root").resolve()
    if path == raw or raw in path.parents:
        raise ValueError(f"Output cannot be inside raw datasets: {path}")
    return path
