# Pomegranate and wheat temporal collections

## Pomegranate

`images/week_01_(13_10_2024)` through `week_14_(12_01_2025)` contain numbered subgroups 01?10. The collection has 2,899 JPEGs and 14 corresponding `weekly_report/*.xlsx` workbooks.

Metadata includes image name, date/time, temperature, dew point, relative humidity, precipitation, radiation, pressure, wind, weather description and visibility. A week-14 sheet header omits `rh`; do not assume identical column positions. No disease/severity target columns were established by the inspection. Repeated filenames need week/group context; filename reuse alone does not prove biological identity.

There is no final loader, split or early-risk target protocol. Future partitioning must respect validated plant/leaf/sequence identities and prediction time boundaries. Random image splitting is not supported.

## Wheat

`data/{2023,2024}/{acquisition date}` contains 418 JPEGs (290 from 2023, 128 from 2024). `content.txt` describes 20 leaf series from 2023 and 10 from 2024 at the ETH research station, and QR-derived filenames containing timestamp, experiment/plot and leaf identifiers.

`models/best.pt` is a supplied reference-mark detector (~23.39 MB). Read-only archive inspection found Ultralytics PoseModel / YOLOv8s-pose references and a Marker class. It is not a FOLTRA-trained crop disease model. Its source marker dataset YAML and original run directory are absent, and exact compatible training environment/provenance remain unresolved. It stays associated with this dataset.

No checkpoint loading or inference occurred. No final temporal training protocol is implemented. Future work must retain leaf identity across acquisition dates and evaluate year/plot/sequence independence.
