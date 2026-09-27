# TOBRFV-LMID: unresolved temporal identity

## Structure

`{RGB,VNIR,VNIR_800nm,VNIR_1000nm}/{Pepper,Tomato}/{I,II}/{Healthy,Virus}/` contains TIFF RGB images and PNG VNIR representations. Each modality has 10,179 files in the inspection snapshot. Fully matched indexes contain 10,178 rows.

## Preserved approaches

| Approach | Source modules | Historical manifests | Meaning |
|---|---|---|---|
| sequence/frame interpretation | `tobrfv_sequence_frame_split` | `tobrfv_train.csv`, `tobrfv_val.csv`, `tobrfv_test.csv` | `H_11_RGB_0_1`: H=status, 11=identifier, 0=capture, 1=frame; grouping is group + H_11 |
| day/angle/pot interpretation | `tobrfv_day_angle_pot_index`, `tobrfv_day_angle_pot_split` | `tobrfv_temporal_index.csv`, `tobrfv_temporal_{train,val,test}.csv` | `Status_Day_Modality_Angle_PotID`; grouping is group + pot |
| general modality alignment | `tobrfv_aligned_index` | `tobrfv_index.csv` | Group + normalized filename alignment; `plant` is crop species, not biological plant identity |

Both splitting approaches use seed 42 and approximate 70/15/15 partitions within experimental groups. The preserved CSVs have not been regenerated. Sequence/frame splits contain 7,087 / 1,572 / 1,520 rows; day/angle/pot splits contain 7,056 / 1,580 / 1,542 rows and 240 / 54 / 53 group-plus-pot sequences.

The day/angle/pot parser recognizes angles 0, 5, 15. Observations sort by day then angle. Multiple angles are observations, not automatically distinct time points. `max_sequence_length` truncates observations, not days. A later experiment must decide how to aggregate views and define temporal sampling.

## Known mismatch

Within `Tomato/I/Virus`, RGB contains `V_20_RGB_15_14.tiff`, while the VNIR modalities have the corresponding normalized key `V_20_15_17` instead of `V_20_15_14`. This explains why equal modality totals yield only 10,178 aligned rows. No file was renamed or corrected.

## Identity and leakage limitations

Filename structure **does not prove biological temporal identity**. The original acquisition documentation is not available locally to establish the canonical interpretation or independence of pots across groups/experiments/statuses. The day/angle/pot approach is a provisional implementation, not a scientifically validated protocol.

Under the day/angle/pot interpretation, the historical sequence/frame train/val, train/test and val/test pairs share 343, 347 and 343 group-plus-pot identities. Those files must not be described as plant-independent temporal splits. Their own grouping-key checks test a different definition.

Day/angle/pot manifests have disjoint constructed group-plus-pot keys, but that does not establish cross-group independence, a future-time holdout, or early detection validity. Status labels alone do not define symptom onset or a forecasting target. Labels, prediction horizon, observation cutoff, causal feature construction and evaluation protocol remain unresolved.

## Loader and augmentation contract

`src.datasets.temporal.TOBRFVTemporalDataset` loads the preserved day/angle/pot format and resolves every image through central configuration. `src.datasets.collate.collate_fn` retains status/group/pot metadata and pads sequences with an explicit boolean mask. It does not invent severity or risk targets.

The generic loader applies a provided transform to individual images. Independent random augmentation must not be used when synchronized multimodal/temporal geometry is required. The default smoke collator only resizes and converts tensors; spectral calibration, channel handling and modality-specific normalization remain unvalidated.

New preparation writes only to `data/manifests/generated/`, refuses existing targets, and requires `--acknowledge-unvalidated` for either semantic split and the temporal index. Canonical protocol selection requires an explicit research decision supported by upstream documentation.
