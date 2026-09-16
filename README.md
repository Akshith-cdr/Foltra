# FOLTRA - AI-Powered Crop Health Monitoring

FOLTRA investigates robust crop-health assessment from images and repeated observations. The research plan includes classification, localization, segmentation, area-based severity, field robustness, temporal modeling and early-risk evaluation.

## Current status

| State | What exists |
|---|---|
| Implemented | Configuration/path utilities, bounded/full dataset audit, class-folder validation, classification splitting/transforms/loaders, TOBRFV indexing, prototype temporal loading/padding, diagnostics and fixture tests |
| Prepared | Fourteen historical CSV manifests for Cassava, PlantVillage color and two incompatible TOBRFV interpretations |
| Planned | ResNet training/evaluation, disease detection models, U-Net, severity estimation, robustness experiments, LSTM/GRU, early-risk prediction, contextual/conversational integration |
| Not yet validated | TOBRFV biological identity/filename semantics, severity denominators/thresholds, early-risk labels/horizons/cutoffs, model performance |

There are no FOLTRA training results or trained disease checkpoints. The week-05 experiment is a planning record. The supplied wheat checkpoint is a dataset-associated reference-marker model, not a FOLTRA disease model.

## Repository structure

```text
configs/                 dataset locations and preprocessing settings
src/datasets/            audit, classification loaders, temporal loader/collator
src/preprocessing/       validation, transforms, split/index implementations
src/utils/               configuration, paths, manifests, preparation safeguards
scripts/data_audit/      explicitly invoked diagnostics
scripts/smoke/           setup and bounded loading checks
tests/                   generated fixtures; no raw datasets required
data/manifests/          preserved CSVs and checksum/provenance record
data/processed/          generated reports/caches, ignored
docs/datasets/           task-specific dataset contracts and limitations
docs/preprocessing/      workflow and command reference
docs/experiments/        experiment-record convention
experiments/week_05_baseline/  planned experiment; no results
Datasets/                local raw data, ignored; setup complete
```

`src` is the current Python package name. Model, training and evaluation plans are documented in `docs/experiments/baseline.md`; unused placeholder modules have been removed.

## Dataset setup

The local dataset collection is now present under `Foltra/Datasets/`. For a new checkout, place the eight directories in this layout:

```text
Datasets/
  casava/
  CD&S/
  plantdoc/
  plantseg/
  plantvillage/
  TOBRFV-LMID/
  pomegranate time series/
  wheat time series/
```

The owner completed the dataset copy. All eight configured dataset directories and 87 representative manifest references were checked successfully; this was not a full integrity audit. Preserve directory spelling and contents, and avoid `Datasets/Datasets/` nesting.

To use the existing sibling collection without copying, manually set `paths.dataset_root: ../Datasets` in `configs/datasets.yaml`. The default is `Datasets`, the current local location. No junction or automatic fallback is used. All manifest `Datasets/...` references are mapped through this setting. No raw file is written by preparation tools.

See [dataset overview](docs/datasets/overview.md) for task-specific layouts. Do not randomly split pomegranate or wheat images. PlantVillage classification uses **color only**; other representations are related views, not independent biological samples.

## Installation

From the repository root, with Python 3.10+ (migration verified on Python 3.11):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

The editable install enables `python -m ...` from other working directories while configuration remains rooted in this source checkout. Use this as a source-checkout project; a standalone wheel without its configuration/data is not supported.

`constraints-observed.txt` records the six actual installed dependency versions observed during migration. To request those direct-dependency versions, add `-c constraints-observed.txt` to the requirements installation command. This is an observed environment snapshot, not a complete transitive or CUDA lockfile. No dependencies were installed during migration.

## Configuration

- `configs/datasets.yaml`: dataset/manifest/processed roots, task types and per-dataset paths.
- `configs/preprocessing.yaml`: seed, ratios, image size, normalization, augmentations and provisional TOBRFV settings.
- Paths resolve from the repository, not the shell's current directory.
- Historical CSVs use portable image-path separators; non-path metadata and membership remain unchanged.

```powershell
python -m scripts.smoke.setup
python -m scripts.data_audit.manifests
python -m unittest discover -s tests -v
python -m scripts.smoke.transforms
```

After raw-data setup, `python -m scripts.data_audit.manifests --check-files` checks referenced filenames without image decoding.

## Dataset audit and validation

A bounded audit explicitly limits image opens and writes three reports into a fresh output directory:

```powershell
python -m src.datasets.audit_datasets --limit 10 --output data/processed/audit-sample
python -m src.preprocessing.validation cassava --limit 10 --verify
```

Audit outputs are `dataset_inventory.csv`, `dataset_summary.json`, and `audit_report.txt`. Counts refer to image files, including masks and alternative representations; they are not unique biological sample counts. HEIC readability depends on an available Pillow decoder.

A full audit is intentionally opt-in and can be expensive:

```powershell
python -m src.datasets.audit_datasets --all --output data/processed/audit-full
```

The full audit was **not** run during migration. Existing reports are not overwritten. Dataset reads are read-only; only the configured report location is written.

## Preprocessing and manifest generation

The preserved manifests in `data/manifests/` are the source for current prepared memberships. Do not regenerate them merely to reproduce directory layout.

For a **new**, separately named classification preparation:

```powershell
python -m src.preprocessing.classification_splits cassava --output-dir data/manifests/new-cassava-run
python -m src.preprocessing.classification_splits plantvillage --output-dir data/manifests/new-plantvillage-run
```

These commands use class stratification, seed 42 and 70/15/15 ratios from configuration. New generation sorts filenames, whereas historical generation did not; identical membership with old CSVs is not promised. Existing targets are refused. Runtime transforms resize/normalize; there is no saved resized-image dataset.

TOBRFV has **two explicitly preserved interpretations**, neither declared the canonical biological temporal protocol. Commands and output locations are documented in [the pipeline guide](docs/preprocessing/pipeline.md). Read [the TOBRFV caveats](docs/datasets/tobrfv.md) before any new preparation. Existing TOBRFV manifests were not regenerated.

## Experiments and reproducibility

Read [reproducibility](docs/reproducibility.md), [migration record](docs/migration/README.md) and [experiment conventions](docs/experiments/baseline.md).

Track configuration, small manifests and honest evaluation records. Raw datasets, processed caches, checkpoint binaries and large generated predictions are ignored. Do not report presentation illustrations as measured results.

There is no training command yet. Canonical temporal identity, organ-area severity definitions, and causal early-risk targets must be resolved before those scientific protocols are implemented.
