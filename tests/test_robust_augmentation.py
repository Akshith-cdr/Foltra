"""Offline checks for the controlled Week 6 augmentation experiment."""
import random
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.evaluation.evaluate_robust_augmentation import metric_block, resolve_plantdoc_root
from src.preprocessing.transforms import RandomJpegCompression, build_transforms
from src.training.train_robust_augmentation import (
    format_duration,
    report_epoch_progress,
    validate_fairness,
)
from src.utils.config import load_config
from scripts.colab.package_week_06 import REQUIRED_CODE_MEMBERS, write_code_archive


class RobustAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config("robust_augmentation")
        self.augmentation = self.config["augmentation"]

    def test_field_style_pipeline_has_exact_order_and_parameters(self):
        pipeline = build_transforms(training=True, augmentation=self.augmentation)
        steps = pipeline.transforms
        self.assertEqual([type(step).__name__ for step in steps], [
            "Lambda", "RandomResizedCrop", "RandomHorizontalFlip", "RandomApply",
            "RandomApply", "RandomJpegCompression", "ToTensor", "Normalize",
        ])
        crop, flip, color, blur, jpeg = steps[1:6]
        self.assertEqual((crop.scale, crop.ratio, crop.size), ((0.8, 1.0), (0.9, 1.1), (224, 224)))
        self.assertEqual(flip.p, 0.5)
        self.assertEqual(color.p, 0.8)
        jitter = color.transforms[0]
        self.assertEqual(jitter.brightness, (0.8, 1.2))
        self.assertEqual(jitter.contrast, (0.8, 1.2))
        self.assertEqual(jitter.saturation, (0.85, 1.15))
        self.assertEqual(jitter.hue, (-0.03, 0.03))
        self.assertEqual(blur.p, 0.15)
        self.assertEqual(blur.transforms[0].kernel_size, (3, 3))
        self.assertEqual(blur.transforms[0].sigma, (0.1, 1.0))
        self.assertEqual((jpeg.probability, jpeg.quality), (0.15, (75, 95)))

    def test_validation_and_test_transform_ignore_training_profile(self):
        image = Image.new("RGB", (31, 19), color=(20, 90, 180))
        baseline = build_transforms(training=False)(image)
        robust_eval = build_transforms(training=False, augmentation=self.augmentation)(image)
        self.assertTrue(torch.equal(baseline, robust_eval))
        self.assertEqual(tuple(robust_eval.shape), (3, 224, 224))
        self.assertEqual([type(step).__name__ for step in build_transforms(False).transforms],
                         ["Lambda", "Resize", "ToTensor", "Normalize"])

    def test_seeded_training_transform_is_reproducible(self):
        image = Image.fromarray(np.arange(96 * 128 * 3, dtype=np.uint8).reshape(96, 128, 3))
        pipeline = build_transforms(training=True, augmentation=self.augmentation)
        random.seed(42); torch.manual_seed(42)
        first = pipeline(image)
        random.seed(42); torch.manual_seed(42)
        second = pipeline(image)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (3, 224, 224))

    def test_jpeg_transform_validates_and_preserves_image_shape(self):
        image = Image.new("RGB", (23, 17), color=(40, 80, 120))
        transformed = RandomJpegCompression(1.0, [80, 80])(image)
        self.assertEqual((transformed.mode, transformed.size), ("RGB", image.size))
        with self.assertRaises(ValueError):
            RandomJpegCompression(1.1, [75, 95])
        with self.assertRaises(ValueError):
            RandomJpegCompression(0.5, [95, 75])

    def test_fairness_controls_match_week_05(self):
        validate_fairness(self.config)
        changed = {**self.config, "training": {**self.config["training"], "epochs": 9}}
        with self.assertRaisesRegex(ValueError, "fairness controls"):
            validate_fairness(changed)

    def test_metric_block_reports_per_class_values(self):
        metrics = metric_block(
            np.asarray([0, 1, 2]), np.asarray([0, 1, 1]), [0, 1, 2],
            ["a", "b", "c"], ["a", "b", "c"], loss=0.25,
        )
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], (1 + 2 / 3) / 3)
        self.assertEqual([row["support"] for row in metrics["per_class"]], [1, 1, 1])
        self.assertEqual(metrics["confusion_matrix"]["counts"], [[1, 0, 0], [0, 1, 0], [0, 1, 0]])

    def test_colab_code_archive_contains_week_06_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "code.zip"
            manifest = write_code_archive(archive_path)
            with ZipFile(archive_path) as archive:
                members = set(archive.namelist())
            self.assertTrue(REQUIRED_CODE_MEMBERS.issubset(members))
            self.assertEqual(set(manifest["required_members_verified"]), REQUIRED_CODE_MEMBERS)
            self.assertIn("week_06_package_manifest.json", members)

    def test_colab_dataset_root_override_is_opt_in(self):
        from src.utils import config
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FOLTRA_DATA_ROOT", None)
            self.assertEqual(config.configured_path("dataset_root"), config.PROJECT_ROOT / "Datasets")
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"FOLTRA_DATA_ROOT": temporary}, clear=False
        ):
            self.assertEqual(config.configured_path("dataset_root"), Path(temporary).resolve())

    def test_robust_evaluator_resolves_plantdoc_from_environment_root(self):
        plantdoc_config = load_config("plantdoc_external_evaluation")
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {"FOLTRA_DATA_ROOT": temporary}, clear=False
        ):
            expected = Path(temporary).resolve() / "plantdoc"
            (expected / "test" / "img").mkdir(parents=True)
            (expected / "test" / "ann").mkdir()
            self.assertEqual(resolve_plantdoc_root(plantdoc_config), expected)
            self.assertTrue((resolve_plantdoc_root(plantdoc_config) / "test" / "img").is_dir())
            self.assertTrue((resolve_plantdoc_root(plantdoc_config) / "test" / "ann").is_dir())

    def test_epoch_progress_reports_metrics_and_measured_timing(self):
        row = {"train_loss": 1.25, "validation_loss": 0.75, "validation_accuracy": 0.625}
        with patch("builtins.print") as output:
            report_epoch_progress(row, epoch=2, total_epochs=10, elapsed=125.0)
        message = output.call_args.args[0]
        self.assertIn("Epoch 2/10 complete", message)
        self.assertIn("train loss 1.250000", message)
        self.assertIn("validation loss 0.750000", message)
        self.assertIn("validation accuracy 62.50%", message)
        self.assertIn("elapsed 00:02:05", message)
        self.assertIn("ETA 00:08:20", message)
        self.assertEqual(format_duration(3661.4), "01:01:01")


if __name__ == "__main__":
    unittest.main()
