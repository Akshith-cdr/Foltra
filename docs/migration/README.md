# Migration record

## Scope and dataset decision

The automated integration preserved the original BTP source/manifests and did not move or delete originals, train models, commit or push changes. Useful code was integrated with configuration and explicit entry points; this was not a wholesale workspace copy.

The owner has completed the raw-data copy into `Foltra/Datasets/`. All eight configured dataset directories and 87 representative manifest references were verified without decoding images. Raw data remains ignored by Git. Copying project/tracker documents and PPT assets was explicitly declined; no manual migration actions remain.

Cleanup removed unused model/training/evaluation scaffolding, empty reserved directories, the obsolete manual-copy checklist and the redundant week-05 status note. Research plans remain in `docs/experiments/baseline.md` and the experiment YAML. This cleanup left files outside Foltra untouched and retained migration provenance.

## Source classification and mapping

A = reusable production/library code; B = diagnostic; C = smoke/test; D = empty/generated/not useful to migrate; E = interpretation requiring reconciliation. Categories may overlap.

| Original under BTP | Category | Destination under Foltra / treatment |
|---|---|---|
| src/preprocessing/validate_classification_dataset.py | A/B | src/preprocessing/validation.py; consolidated validation/property inspection |
| src/preprocessing/inspect_image_properties.py | A/B | src/preprocessing/validation.py |
| src/preprocessing/split_classification_dataset.py | A | src/preprocessing/classification_splits.py; central config, sorted new generation, explicit fresh output |
| src/preprocessing/transforms.py | A | src/preprocessing/transforms.py; config-backed; top-level RGB callable |
| src/preprocessing/verify_split.py | B | scripts/data_audit/verify_split.py |
| src/preprocessing/build_tobrfv_index.py | A/E | src/preprocessing/tobrfv_aligned_index.py |
| src/preprocessing/build_tobrfv_temporal_index.py | A/E | src/preprocessing/tobrfv_day_angle_pot_index.py |
| src/preprocessing/split_tobrfv_sequences.py | A/E | src/preprocessing/tobrfv_sequence_frame_split.py |
| src/preprocessing/split_tobrfv_temporal.py | A/E | src/preprocessing/tobrfv_day_angle_pot_split.py |
| src/preprocessing/analyze_tobrfv_groups.py | B/E | scripts/data_audit/analyze_tobrfv_groups.py; species/plant warning added |
| src/preprocessing/analyze_tobrfv_sequences.py | B/E | scripts/data_audit/analyze_tobrfv_sequences.py |
| src/preprocessing/verify_tobrfv_identifiers.py | B/E | scripts/data_audit/verify_tobrfv_identifiers.py |
| src/preprocessing/verify_tobrfv_modalities.py | B | scripts/data_audit/verify_tobrfv_modalities.py; import-time scanning removed |
| src/preprocessing/verify_tobrfv_temporal_structure.py | B/E | scripts/data_audit/verify_tobrfv_temporal_structure.py |
| src/data/tobrfv_temporal_dataset.py | A/E | src/datasets/temporal.py; repository/config-root paths |
| src/preprocessing/test_tobrfv_dataloader.py | A/C | reusable helpers extracted to src/datasets/collate.py; bounded scripts/smoke/temporal.py; fixture coverage |
| src/preprocessing/test_tobrfv_temporal_dataset.py | C | scripts/smoke/temporal.py consolidates loading smoke behavior |
| src/preprocessing/test_transforms.py | C | scripts/smoke/transforms.py uses generated in-memory pixels |
| Datasets/plantdoc_stats.py | B | scripts/data_audit/plantdoc_stats.py; no work on import |
| Datasets/tobrfv_stats.py | B | scripts/data_audit/tobrfv_stats.py; no work on import |
| src/datasets/audit_datasets.py | D | Empty original left intact; not copied over working Foltra audit |
| src/**/__pycache__ | D | Not migrated |
| data/splits/*.csv | Prepared artifacts/E for TOBRFV | data/manifests/; 14 files, portable path fields only |

The original Foltra audit was replaced with a streaming report writer, bounded sampling, no import-time writes, no machine paths, and raw-root output protection. Cassava's existing placeholder now delegates to a reusable manifest classification loader; PlantVillage has the same explicit color-manifest adapter. Unused model/training/evaluation placeholders were subsequently removed; their plans remain in the experiment documentation.

Both TOBRFV preparation approaches retain their original algorithms. No original implementation was deleted and no TOBRFV split was regenerated. Existing sequence identifiers/group strings remain unchanged in migrated CSVs. New outputs are separate and protected from overwrite.

## Intentionally not migrated

- No automated raw-data copy was performed. The owner subsequently copied the dataset collection, including its associated wheat reference checkpoint.
- Python caches; empty BTP audit/model/training/evaluation scaffolding and empty configs/models/experiments roots.
- Administrative resume/consent documents and unrelated materials.
- PPT/PDF/PNG duplicates and the project/tracker binaries. They remain available in BTP. Existing repository presentation/license remain unchanged. Research decisions are summarized in repository documentation; no binary consolidation was needed for pipeline migration.

## Verification commands

```powershell
python -B -m unittest discover -s tests -v
python -B -m scripts.smoke.migration_integrity
python -B -m scripts.data_audit.manifests
python -B -m scripts.smoke.setup
python -B -m scripts.smoke.transforms
```

`--originals` additionally checks the historical BTP locations and requires those originals to remain there. After the owner's dataset migration, `BTP/Datasets/plantdoc_stats.py` and `BTP/Datasets/tobrfv_stats.py` are absent from those historical locations; both are present under `Foltra/Datasets/` with their original hashes. The remaining original-file checksums match. Use the default command above for the completed repository, including standalone clones. Original hashes and destination hashes document preservation; hashes do not establish semantic label quality.

The migration is complete. Its superseded manual-copy checklist has been removed. No scientific protocol choice, full audit, model training or raw dataset modification is included in this migration.
