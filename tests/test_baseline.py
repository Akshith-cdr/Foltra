"""Offline checks of classifier shape, metrics and artifacts."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.resnet_baseline import build_model
from src.evaluation.evaluate import evaluate, save_metrics


class BaselineTests(unittest.TestCase):
    def test_classification_resolves_root_once(self):
        from src.datasets.classification import ClassificationDataset
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (8, 8)).save(root / "image.png")
            with patch("src.datasets.classification.load_manifest", return_value=[
                {"path": "Datasets/image.png", "label": "healthy"}
            ]), patch("src.datasets.classification.resolve_image_path", return_value=root) as resolve:
                dataset = ClassificationDataset("fixture.csv")
                self.assertEqual(dataset[0][1], 0)
                self.assertEqual(dataset[0][0].size, (8, 8))
                resolve.assert_called_once_with("Datasets")

    def test_classifier_matches_classes(self):
        model = build_model(7, pretrained=False)
        self.assertEqual(model.fc.out_features, 7)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_metrics_and_labeled_matrix(self):
        logits = torch.tensor([[4., 0., 0.], [0., 4., 0.], [0., 4., 0.]])
        labels = torch.tensor([0, 1, 2])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        result = evaluate(torch.nn.Identity(), loader, torch.device('cpu'), 3)
        self.assertAlmostEqual(result['accuracy'], 2 / 3)
        self.assertAlmostEqual(result['macro_f1'], (1 + 2 / 3) / 3)
        self.assertAlmostEqual(result['loss'], torch.nn.functional.cross_entropy(logits, labels).item(), places=6)
        self.assertEqual(result['confusion_matrix'], [[1, 0, 0], [0, 1, 0], [0, 1, 0]])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'test'
            save_metrics(result, {'c': 2, 'a': 0, 'b': 1}, output)
            with (output / 'confusion_matrix.csv').open(newline='') as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], ['true / predicted', 'a', 'b', 'c'])
            with self.assertRaises(FileExistsError):
                save_metrics(result, {'a': 0}, output)
