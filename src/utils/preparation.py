"""Safety checks shared by preserved preparation entry points."""
import argparse
from pathlib import Path
from .config import ensure_output_path, configured_path


def run_preparation(main, outputs, unvalidated=False):
    parser = argparse.ArgumentParser(description="Explicit preparation run; historical manifests are preserved")
    if unvalidated:
        parser.add_argument("--acknowledge-unvalidated", action="store_true", required=True,
                            help="This filename interpretation is not a canonical biological protocol")
    parser.parse_args()
    if not configured_path("dataset_root").is_dir():
        raise FileNotFoundError("Dataset root missing. See README Dataset setup.")
    check_outputs(outputs)
    main()


def check_outputs(outputs):
    for path in outputs:
        target = ensure_output_path(path)
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}; review previous generated outputs first")
