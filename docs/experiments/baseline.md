# Week 5 PlantVillage Color baseline

The baseline fine-tunes all layers of ImageNet-pretrained torchvision ResNet-18.
Class mapping comes from the complete training manifest and is shared with val/test.
Existing manifests are read unchanged. No full training has been run.

From the Foltra repository root:

```powershell
python -B -m src.training.train_baseline --config baseline --smoke
```

Smoke mode uses one epoch, eight seeded random samples per split, batches of four,
and the full class mapping. Its metrics verify execution, not model quality.
The first run downloads pretrained weights if needed; it never silently falls back
to random initialization. Omit `--smoke` for full training when ready.

Settings live in `configs/baseline.yaml`: epochs, batch size, learning rate,
weight decay, workers, CPU threads, device, seed, manifests and output root.
`--config NAME` selects another YAML under configs. Existing preprocessing settings
control RGB conversion, resize, ImageNet normalization and training flip/rotation.
Validation/test transforms are deterministic. Seeds control sampling and shuffling;
bitwise equivalence across different hardware is not guaranteed.

Unique run folders under `experiments/week_05_baseline/outputs/` contain:

- `run.json`: effective config, preprocessing, class mapping, manifest hashes,
  sample counts, runtime details and completion/failure status.
- `history.json`: sample-weighted train loss, validation loss and accuracy per epoch.
- `best.pt`: model/optimizer states and metadata for the lowest validation loss.
- `test/metrics.json`: test loss, accuracy, macro F1 and confusion matrix.
- `test/confusion_matrix.csv`: labeled true-class rows and predicted-class columns.

Test evaluation uses the best validation checkpoint after training. Macro F1
includes every training class, using zero for undefined scores. Accuracy is a
fraction from zero to one. All run artifacts and the shared pretrained cache
inside outputs are ignored by Git.

Re-evaluate a saved checkpoint without downloading pretrained weights:

```powershell
python -B -m src.evaluation.evaluate --checkpoint experiments/week_05_baseline/outputs/<run>/best.pt
```

Evaluation verifies manifest hashes and preprocessing against the checkpoint,
preserves its smoke/full test scope, and writes a fresh evaluation directory.

Reference: [torchvision ResNet-18 implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py).

Training prints the selected device and sample counts, then batch progress and an estimated remaining time for each phase. `training.log_every_batches` controls reporting frequency (default 20); the first and last batches are always reported. Estimates include loading and computation and settle as more batches complete. Image paths are resolved once per dataset to avoid repeated configuration reads. CPU training still requires substantial computation; these changes do not reduce epochs, image size, or training samples.
