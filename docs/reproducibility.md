# Reproducibility and limits

## Preserved evidence

`data/manifests/provenance.json` records original and migrated CSV hashes, schemas, row counts and generation assumptions. Only file-reference separators changed from backslash to slash. Membership, row order, labels, group strings and sequence IDs remain unchanged. Historical files under BTP were not overwritten.

`docs/migration/original_checksums.json` records SHA-256 hashes of original BTP source, manifests and cache files plus the two dataset utilities. Verification compares those exact source files and checks they still exist. This is not a raw-image content checksum catalog.

Prepared classifications: stratified 70/15/15, seed 42, PlantVillage color. Prepared TOBRFV: approximate within-group 70/15/15, seed 42, two incompatible identity interpretations. Preserve generated CSVs because historic filesystem enumeration was not sorted. New classification generation sorts paths but may differ from historical membership.

## Environment

Python 3.11 was used for migration checks. `requirements.txt` lists actual imported packages, including PyYAML for centralized configuration. `constraints-observed.txt` and `docs/migration/observed_environment.json` record observed direct dependency versions. These are not a full transitive lock or GPU/CUDA specification. Record Python, OS, accelerator/runtime and all package versions for each future experiment. The wheat reference checkpoint may require a separate historical Ultralytics environment; it is not loaded by core code.

## Lightweight validation

`python -m unittest discover -s tests -v` uses synthetic tiny images/manifests and no downloads or real dataset scan. It checks root-independent paths, configuration, portable manifests, bounded audit reports/corrupt fixtures, deterministic classification partitioning, import side effects, the presence of both TOBRFV parsers and temporal padding/metadata.

`python -m scripts.data_audit.manifests` validates historical CSV schemas and path syntax without opening images. Add `--check-files` only after configuring raw-data access. `scripts.smoke.setup` explicitly reports a missing raw root; it does not silently fall back to BTP.

## Scientific limits

No trained model, metric, severity percentage, early-risk result or fully validated biological sequence protocol was created. Whole-plant/leaf identity, duplicate imagery, PlantVillage variants, cross-group independence and time-causal evaluation need research-specific checks. Status labels do not establish onset or prediction horizon. Random per-image augmentations are not synchronized temporal/multimodal transforms.

## Tracking policy

Track source, configuration, documentation, small manifests, provenance and small evaluated metrics. Ignore raw datasets, processed caches, checkpoints, run directories and large predictions. Dataset documentation and licensing must be maintained separately from the software MIT license. Preserve original BTP work until a later explicit cleanup decision; migration did not authorize deletion.
