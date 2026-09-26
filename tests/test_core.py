"""Small synthetic fixtures only; no real dataset enumeration or training."""
import csv
import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from src.utils import config
from src.utils.manifests import load_manifest, validate_manifest
from src.datasets.audit_datasets import audit
from src.preprocessing.classification_splits import split_samples


class CoreTests(unittest.TestCase):
    def test_root_independent_of_cwd(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                self.assertEqual(config.project_path("README.md"), config.PROJECT_ROOT / "README.md")
                self.assertEqual(config.load_config()["paths"]["manifest_root"], "data/manifests")
                self.assertEqual(config.configured_path("dataset_root"), config.PROJECT_ROOT / "Datasets")
            finally:
                os.chdir(previous)

    def test_configuration(self):
        cfg = config.load_config("preprocessing")
        self.assertAlmostEqual(sum(cfg["classification"][k+"_ratio"] for k in ("train", "val", "test")), 1)
        self.assertEqual(config.load_config()["datasets"]["plantvillage"]["representation"], "color")
        self.assertEqual(cfg["tobrfv"]["canonical_protocol"], "unresolved")

    def test_portable_path_rejection(self):
        self.assertEqual(config.portable_path("Datasets\\casava\\x.jpg"), "Datasets/casava/x.jpg")
        for path in ("C:/private/x.jpg", "/tmp/x.jpg", "../Datasts/x.jpg", "Datasets/../../x.jpg"):
            with self.assertRaises(ValueError):
                config.portable_path(path)

    def test_manifest_schema_and_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); raw = root / "raw"; raw.mkdir()
            (raw / "x.jpg").write_bytes(b"existence only")
            manifest = root / "samples.csv"
            manifest.write_text("path,label\nDatasets/x.jpg,healthy\n", encoding="utf-8")
            with patch("src.utils.config.configured_path", return_value=raw):
                self.assertEqual(validate_manifest(manifest, ["path", "label"], True)["missing"], [])
            with self.assertRaises(ValueError):
                load_manifest(manifest, ["sequence_id"])
            manifest.write_text("path,label\n,healthy\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_manifest(manifest)

    def test_small_audit_and_raw_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); raw = root / "raw"; images = raw / "example" / "train"; images.mkdir(parents=True)
            Image.new("RGB", (9, 7)).save(images / "good.png")
            (images / "broken.png").write_bytes(b"not an image")
            before = {p.name:p.read_bytes() for p in images.iterdir()}
            summary = audit(raw, root / "reports", limit=2)
            self.assertEqual(summary["total_images"], 2)
            self.assertEqual(summary["image_status"], {"unreadable": 1, "ok": 1})
            self.assertEqual(summary["resolutions"], {"9x7": 1})
            self.assertEqual(summary["splits"], {"train": 2})
            self.assertEqual(before, {p.name:p.read_bytes() for p in images.iterdir()})
            for name in ("dataset_inventory.csv", "dataset_summary.json", "audit_report.txt"):
                self.assertTrue((root / "reports" / name).is_file())
            with self.assertRaises(ValueError):
                audit(raw, raw / "reports", limit=1)
            with self.assertRaises(FileExistsError):
                audit(raw, root / "reports", limit=1)
            self.assertEqual(audit(raw, root / "one", limit=1)["total_images"], 1)

    def test_classification_membership_and_determinism(self):
        samples = [{"path": f"Datasets/example/{label}/{i}.png", "label": label} for label in ("a", "b") for i in range(40)]
        splits = split_samples(samples)
        self.assertEqual(splits, split_samples(samples))
        sets = [{row["path"] for row in rows} for rows in splits.values()]
        self.assertEqual(len(set.union(*sets)), len(samples))
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertTrue(all({r["label"] for r in rows} == {"a", "b"} for rows in splits.values()))

    def test_imports_have_no_dataset_scans(self):
        names = ["src.datasets.temporal", "src.datasets.collate", "src.datasets.cassava", "src.datasets.plantvillage",
                 "src.preprocessing.transforms", "src.preprocessing.validation", "src.preprocessing.tobrfv_aligned_index",
                 "src.preprocessing.tobrfv_day_angle_pot_index", "src.preprocessing.tobrfv_day_angle_pot_split",
                 "src.preprocessing.tobrfv_sequence_frame_split", "scripts.data_audit.verify_tobrfv_modalities",
                 "scripts.data_audit.plantdoc_stats", "scripts.data_audit.tobrfv_stats"]
        # Heavy libraries initialize before blocking filesystem enumeration.
        import torch, torchvision, pandas, sklearn
        with patch("os.walk", side_effect=AssertionError("Unexpected dataset scan on import")):
            for name in names:
                importlib.import_module(name)

    def test_both_tobrfv_interpretations_retained(self):
        from src.preprocessing.tobrfv_day_angle_pot_index import parse_filename
        from src.preprocessing.tobrfv_sequence_frame_split import PATTERN
        parsed = parse_filename("H_11_RGB_0_1.tiff")
        self.assertEqual((parsed["day"], parsed["angle"], parsed["pot_id"]), (11, 0, 1))
        match = PATTERN.match("H_11_RGB_0_1.tiff")
        self.assertEqual((match.group(2), match.group(4)), ("11", "1"))
        self.assertIsNone(parse_filename("not_a_sample.png"))

    def test_temporal_loading_and_padding(self):
        from src.datasets.temporal import TOBRFVTemporalDataset
        from src.datasets.collate import collate_fn
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); image = root / "sample.png"
            Image.new("RGB", (8, 8)).save(image)
            fields = ["group", "status", "day", "angle", "pot_id", "sequence_id", "rgb_path", "vnir_path", "vnir_800nm_path", "vnir_1000nm_path"]
            rows = []
            for pot, days in [(1, [2, 1]), (2, [1])]:
                for day in days:
                    rows.append(dict(group="test", status="H", day=day, angle=0, pot_id=pot, sequence_id=f"test|Pot_{pot}", **{key:"Datasets/sample.png" for key in fields if key.endswith("_path")}))
            path = root / "temporal.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            with patch("src.datasets.temporal.resolve_image_path", return_value=image):
                dataset = TOBRFVTemporalDataset(path)
                self.assertEqual(dataset[0]["days"].tolist(), [1, 2])
                batch = collate_fn([dataset[0], dataset[1]])
                self.assertEqual(batch["lengths"].tolist(), [2, 1])
                self.assertEqual(batch["mask"].tolist(), [[True, True], [True, False]])
                self.assertEqual(batch["statuses"], ["H", "H"])
                self.assertEqual(batch["pot_ids"], [1, 2])
                self.assertEqual(tuple(batch["rgb"].shape), (2, 2, 3, 224, 224))


if __name__ == "__main__":
    unittest.main()
