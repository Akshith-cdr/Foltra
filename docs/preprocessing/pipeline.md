# Implemented preprocessing workflow

All commands are run from the Foltra checkout, or after its editable install. Configuration always resolves relative to the checkout. Raw inputs are never written. Audits and diagnostics are explicit entry points; importing modules does not scan datasets or create output directories.

## Classification

`casava/<class>/<image>` or `plantvillage/color/<class>/<image>` -> optional explicit validation -> stratified CSV generation -> runtime image transforms -> manifest dataset loader.

- `python -m src.preprocessing.validation cassava --limit 10 --verify`
- `python -m src.preprocessing.classification_splits cassava --output-dir data/manifests/new-cassava-run`
- `python -m src.preprocessing.classification_splits plantvillage --output-dir data/manifests/new-plantvillage-run`
- `python -m scripts.data_audit.verify_split casava`

`ClassificationDataset` exposes an optional shared `class_to_idx`; pass the training mapping to validation/test when using a subset with incomplete class coverage. Existing prepared manifests have all classes in each split. No training loop exists.

New splits sort input paths for deterministic generation. Existing split membership was preserved instead of regenerated: historical enumeration ordering is not reproducible from seed alone.

## TOBRFV preparation (explicit, not run during migration)

- `python -m src.preprocessing.tobrfv_aligned_index` -> `data/manifests/generated/tobrfv_index.csv`
- `python -m src.preprocessing.tobrfv_day_angle_pot_index --acknowledge-unvalidated` -> `data/manifests/generated/tobrfv_temporal_index.csv`
- `python -m src.preprocessing.tobrfv_sequence_frame_split --acknowledge-unvalidated` -> `data/manifests/generated/tobrfv_{train,val,test}.csv`
- `python -m src.preprocessing.tobrfv_day_angle_pot_split --acknowledge-unvalidated` -> `data/manifests/generated/tobrfv_temporal_{train,val,test}.csv`

The day/angle/pot splitter deliberately reads the **preserved** `data/manifests/tobrfv_temporal_index.csv`; a newly generated index does not silently replace its input. Review an index before changing this explicit preparation input. Output filenames cannot already exist.

These modules retain the distinct original algorithms. They are preparation utilities, not a canonical research protocol. See [TOBRFV semantics](../datasets/tobrfv.md). Input filename conventions are dataset-specific; arbitrary modalities/angles are not interchangeable acquisition formats.

## Diagnostics

Under `scripts.data_audit`: `manifests`, `analyze_tobrfv_groups`, `analyze_tobrfv_sequences`, `verify_tobrfv_identifiers`, `verify_tobrfv_modalities`, `verify_tobrfv_temporal_structure`, `plantdoc_stats`, `tobrfv_stats`, `verify_split`.

Several TOBRFV diagnostics retain the historical sequence/frame interpretation. Their printed plant/sequence statistics must be read with the documented semantic warnings. Some scan filename trees or annotations; they are not unit tests. `verify_split` checks classification path overlap, not pixel similarity or biological independence.

## Smoke checks

`python -m scripts.smoke.setup` only inspects configured root existence.

`python -m scripts.smoke.transforms` uses in-memory generated pixels.

`python -m scripts.smoke.temporal` explicitly opens at most two observations from two historical sequences (four modalities each). Run it only after raw-data setup. It does not train a model.

## Audit reports

`python -m src.datasets.audit_datasets --limit 10 --output data/processed/audit-sample` writes a bounded report. `--all` instead opts into a full image verification run. The audit distinguishes PlantSeg masks from other image files, retains folder context rather than guessing crop/disease labels, and labels unreadable files without modifying them.
