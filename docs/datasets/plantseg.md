# PlantSeg

Keep four distinct assets: `images/{train,test}` (JPEGs), `annotations/{train,test}` (PNG masks), `json/{train,test}` (LabelMe polygons), and `Metadata.csv`.

The inspection found 9,163 train and 2,295 test image-mask pairs. All external images have matching mask and JSON stems. JSON counts are 9,182 train and 2,298 test: 19 train and 3 test JSON files have no same-stem external image. The sample JSON also embeds image bytes and references an `imagePath` different from the external filename; do not blindly join using that field.

Metadata has crop, disease, resolution, mask filename, mask ratio, source URL and train/test assignment. Its mask ratio is **not automatically organ-relative disease severity**. The denominator, lesion interpretation, organ masks and disease-specific grading must be established before implementing severity calculations.

Unmatched JSON stems from the inspection:

- Train: `blueberry_rust_google_0012`; `zucchini_bacterial_wilt_Bing_0008`, `_0034`, `_0041`, `_0042`, `_0044`, `_0046`, `_0055`; `zucchini_bacterial_wilt_Google_0002`, `_0007`, `_0009`; `zucchini_downy_mildew_Bing_0110`; `zucchini_powdery_mildew_Bing_0035`, `_0061`, `_0071`, `_0119`, `_0220`; `zucchini_powdery_mildew_Google_0105`, `_0343`.
- Test: `raspberry_leaf_spot_Bing_0009`, `zucchini_bacterial_wilt_Bing_0078`, `zucchini_powdery_mildew_Bing_0245`.

Suffix-only entries above inherit the immediately preceding full class/search-source prefix. No annotations were deleted, regenerated or repaired. A training segmentation adapter is planned, not implemented.
