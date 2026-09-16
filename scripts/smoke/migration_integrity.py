"""Verify migration provenance without opening raw images or regenerating data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from src.utils.config import PROJECT_ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--originals", action="store_true", help="Also compare still-preserved BTP sources/manifests")
    args = parser.parse_args()
    records = json.loads((PROJECT_ROOT / "data/manifests/provenance.json").read_text())
    for record in records["files"]:
        destination = PROJECT_ROOT / record["destination"]
        if digest(destination) != record["destination_sha256"]:
            raise ValueError(f"Migrated manifest changed: {destination}")
        if args.originals:
            source = PROJECT_ROOT.parent / Path(record["source"]).relative_to("BTP")
            with source.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle); fields = reader.fieldnames; original = list(reader)
            for row in original:
                for key in fields:
                    if key == "path" or key.endswith("_path"):
                        row[key] = row[key].replace("\\", "/")
            with destination.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if fields != reader.fieldnames or original != list(reader):
                    raise ValueError(f"Schema, membership or metadata changed: {destination}")
    print(f"Verified {len(records['files'])} migrated manifest checksums")
    if args.originals:
        originals = json.loads((PROJECT_ROOT / "docs/migration/original_checksums.json").read_text())
        for name, expected in originals.items():
            source = PROJECT_ROOT.parent / name
            if not source.is_file() or digest(source) != expected:
                raise ValueError(f"Original missing or changed: {source}")
        print(f"Verified {len(originals)} original BTP files unchanged; manifest row membership/order preserved")
    count = 0
    for directory in ("src", "scripts", "tests"):
        for path in (PROJECT_ROOT / directory).rglob("*.py"):
            compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
            count += 1
    print(f"Compiled {count} Python sources in memory (no bytecode output)")


if __name__ == "__main__":
    main()
