# CD&S

Preserve `dataset original/{train,test}/{gls,nlb,nls}`, `dataset annotated/train/{gls,nlb,nls}`, and `dataset severity/{one,two,three,four,five}` separately.

The inspection found 786 original train images and 785 test images, 731 annotated-branch images (including seven HEIC files), and 1,077 severity images. These are branch counts, not claims of unique source images. Original/annotated branches may share identities and must not be split independently.

Annotation text uses YOLO-style rows: class ID, normalized center x/y, width/height. The inspected `classes.txt` order is `nlb`, `nls`, `gls`; do not infer class IDs from alphabetical directory order. Preserve per-folder class definitions and verify consistency before training.

Severity is ordinal folder information. Exact percentage thresholds and clinical meaning are not determined from the current repository. No percentage mapping or severity calculation is implemented. HEIC support depends on an installed decoder; no decoder was installed during migration.
