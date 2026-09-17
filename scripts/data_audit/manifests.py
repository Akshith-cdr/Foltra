"""Validate manifest schema and portability; optionally check referenced files."""
import argparse
from src.utils.config import configured_path
from src.utils.manifests import validate_manifest


def required_columns(name):
    if name.startswith(("casava_", "plantvillage_")):
        return ["path", "label"]
    paths = ["rgb_path", "vnir_path", "vnir_800nm_path", "vnir_1000nm_path"]
    if name == "tobrfv_index.csv":
        return ["sample_id", "group", "label", "plant", "experiment"] + paths
    if name.startswith("tobrfv_temporal_"):
        return ["group", "status", "pot_id", "day", "angle"] + paths + ([] if name.endswith("index.csv") else ["sequence_id", "split"])
    return ["split", "group", "plant_type", "sequence_id", "identifier", "capture", "frame", "rgb_path"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    paths = sorted(configured_path("manifest_root").glob("*.csv"))
    if not paths:
        raise FileNotFoundError("No manifests found")
    for path in paths:
        result = validate_manifest(path, required_columns(path.name), args.check_files)
        print(path.name, "rows", result["rows"], "missing", len(result["missing"]))
        if result["missing"]:
            raise FileNotFoundError(result["missing"][:3])


if __name__ == "__main__":
    main()
