"""Offline checks for PlantDoc object-crop external evaluation."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.evaluation.evaluate_plantdoc_external import (
    PlantDocObjectDataset,
    compute_metrics,
    resolve_plantdoc_root,
    validate_expected_subset,
)
from src.utils.config import load_config


class PlantDocExternalTests(unittest.TestCase):
    def test_evaluator_resolves_environment_dataset_root(self):
        config = load_config("plantdoc_external_evaluation")
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"FOLTRA_DATA_ROOT": temporary}, clear=False
        ):
            root = Path(temporary).resolve() / "plantdoc"
            (root / "test" / "img").mkdir(parents=True)
            (root / "test" / "ann").mkdir()
            resolved = resolve_plantdoc_root(config)
            self.assertEqual(resolved, root)
            dataset = PlantDocObjectDataset(
                resolved, "test", {"approved": "target"}, {"target": 0}, None
            )
            self.assertEqual(dataset.root, root)

    def test_approved_rectangles_become_crops(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "test" / "ann").mkdir(parents=True)
            (root / "test" / "img").mkdir()
            Image.new("RGB", (10, 8), color="red").save(root / "test" / "img" / "sample.jpg")
            annotation = {
                "size": {"width": 10, "height": 8},
                "objects": [
                    {"classTitle": "approved", "geometryType": "rectangle",
                     "points": {"exterior": [[1, 2], [7, 6]]}},
                    {"classTitle": "excluded", "geometryType": "rectangle",
                     "points": {"exterior": [[0, 0], [2, 2]]}},
                ],
            }
            (root / "test" / "ann" / "sample.jpg.json").write_text(json.dumps(annotation), encoding="utf-8")
            dataset = PlantDocObjectDataset(root, "test", {"approved": "target"}, {"target": 3}, None)
            crop, target, index = dataset[0]
            self.assertEqual((crop.size, target, index), ((6, 4), 3, 0))
            self.assertEqual(len(dataset.source_images), 1)
            validate_expected_subset(dataset, {"source_images": 1, "crops": 1, "classes": 1})

    def test_expected_subset_mismatch_stops(self):
        class Fixture:
            source_images = {"one"}
            plantdoc_classes = {"a"}

            def __len__(self):
                return 2

        with self.assertRaisesRegex(ValueError, "refusing inference"):
            validate_expected_subset(Fixture(), {"source_images": 1, "crops": 3, "classes": 1})

    def test_exif_orientation_matches_annotation_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "test" / "ann").mkdir(parents=True)
            (root / "test" / "img").mkdir()
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (10, 8), color="blue").save(
                root / "test" / "img" / "oriented.jpg", exif=exif
            )
            annotation = {
                "size": {"width": 8, "height": 10},
                "objects": [{
                    "classTitle": "approved", "geometryType": "rectangle",
                    "points": {"exterior": [[1, 2], [7, 9]]},
                }],
            }
            (root / "test" / "ann" / "oriented.jpg.json").write_text(
                json.dumps(annotation), encoding="utf-8"
            )
            dataset = PlantDocObjectDataset(root, "test", {"approved": "target"}, {"target": 0}, None)
            crop, _, _ = dataset[0]
            self.assertEqual(crop.size, (6, 7))

    def test_metrics_cover_only_approved_true_classes(self):
        metrics = compute_metrics(
            targets=[1, 1, 3], predictions=[1, 0, 0],
            evaluation_indices=[1, 3], evaluation_names=["one", "three"],
            all_names=["zero", "one", "two", "three"],
        )
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], 1 / 3)
        self.assertEqual(metrics["correct_predictions"], 1)
        self.assertEqual(metrics["confusion_matrix"]["counts"], [[1, 1, 0, 0], [1, 0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
