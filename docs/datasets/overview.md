# Dataset contracts

Counts are the inspection snapshot, not a newly executed image-integrity audit. All raw locations are relative to `paths.dataset_root`. The task registry in `configs/datasets.yaml` describes dataset capabilities; it does not claim a training adapter/model for every task.

| Dataset | Local organization | Task / prepared status |
|---|---|---|
| Cassava (`casava`) | Five class folders; 21,397 JPEGs | Classification; 14,977 / 3,210 / 3,210 prepared memberships |
| PlantVillage | color/grayscale/segmented, 38 classes | Color classification: 38,013 / 8,146 / 8,146 memberships; variants are not independent sample pools |
| CD&S | original train/test, annotated train, severity one?five | Classification, boxes, ordinal severity; no FOLTRA training loader/protocol |
| PlantDoc | train/test with img/ann; meta.json | Detection, 2,251 / 231 images; Supervisely rectangle annotations, 29 metadata classes |
| PlantSeg | images, annotations (masks), json, Metadata.csv | Segmentation; 9,163 / 2,295 image-mask pairs |
| TOBRFV-LMID | Four modalities, crop/experiment/status groups | 10,178 aligned observations; competing historical grouping semantics |
| Pomegranate | 14 dated weeks, numbered groups, weekly XLSX reports | 2,899 images; temporal/context candidate; no disease-target protocol |
| Wheat | year/date acquisitions, experiment/plot/leaf filenames | 418 images; documentation describes 30 leaf series; reference checkpoint supplied |

Cassava labels: bacterial blight, brown streak disease, green mottle, healthy, mosaic disease. Mosaic has 13,158 images: preserve class imbalance in reporting. The on-disk spelling `casava` and historical CSV prefix remain unchanged.

PlantVillage has 54,305 color, 54,305 grayscale and 54,306 segmented files. Only color is supported by the current classification splitter. Cross-representation sample identity must be reconciled before variants enter experiments.

PlantDoc JSON contains potentially multiple rectangle objects per image (`objects`, `classTitle`, `points.exterior`). `meta.json` provides the class definitions. `README.md` and `LICENSE.md` identify its detection purpose and CC BY 4.0 notice. The migrated `plantdoc_stats` diagnostic reads these annotations; it does not convert them to a single-label classification dataset.

See [TOBRFV](tobrfv.md), [PlantSeg](plantseg.md), [CD&S](cds.md) and [temporal datasets](temporal_collections.md) for additional contracts. Complete upstream versions, acquisition URLs and licenses for all collections are not determined from the current repository. Preserve dataset-supplied notices and establish provenance before redistribution.
