"""Migrated split_tobrfv_temporal.py. Filename semantics remain unresolved; see docs/datasets/tobrfv.md."""
from src.utils.config import dataset_path, manifest_path, load_config, image_reference
from src.utils.preparation import run_preparation, check_outputs
import os
import random
import pandas as pd


INPUT_FILE = str(manifest_path("tobrfv_temporal_index.csv"))
OUTPUT_DIR = str(manifest_path("generated"))

SEED = load_config("preprocessing")["random_seed"]
TRAIN_RATIO = load_config("preprocessing")["tobrfv"]["train_ratio"]
VAL_RATIO = load_config("preprocessing")["tobrfv"]["val_ratio"]
TEST_RATIO = load_config("preprocessing")["tobrfv"]["test_ratio"]


def main():
    check_outputs([os.path.join(OUTPUT_DIR, "tobrfv_temporal_" + split + ".csv") for split in ("train", "val", "test")])
    print("TOBRFV-LMID TEMPORAL SPLITTER")
    print("=" * 75)
    print(f"Random seed: {SEED}")
    print("Target split: 70% / 15% / 15%")
    print()

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Temporal index not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("Temporal index is empty.")

    print(f"Loaded observations: {len(df)}")

    required_columns = [
        "group",
        "status",
        "day",
        "angle",
        "pot_id",
        "rgb_path",
        "vnir_path",
        "vnir_800nm_path",
        "vnir_1000nm_path",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    # ------------------------------------------------------------
    # Create a unique plant/pot trajectory identifier.
    #
    # Pot IDs repeat across groups, so group + pot_id is required.
    # ------------------------------------------------------------
    df["sequence_id"] = (
        df["group"].astype(str)
        + "|Pot_"
        + df["pot_id"].astype(str)
    )

    sequences = (
        df[["sequence_id", "group", "pot_id"]]
        .drop_duplicates()
        .to_dict("records")
    )

    print(f"Unique plant/pot sequences: {len(sequences)}")
    print()

    # ------------------------------------------------------------
    # Verify each sequence belongs to exactly one group.
    # ------------------------------------------------------------
    group_check = df.groupby("sequence_id")["group"].nunique()

    bad_sequences = group_check[group_check > 1]

    if len(bad_sequences) > 0:
        print("ERROR: Some sequences belong to multiple groups:")
        for seq in bad_sequences.index:
            print(f"  {seq}")
        raise ValueError("Invalid sequence/group structure.")

    # ------------------------------------------------------------
    # Stratified sequence-level split.
    #
    # We split independently inside each group so that the eight
    # experimental groups remain represented in every split.
    # ------------------------------------------------------------
    random.seed(SEED)

    train_sequences = []
    val_sequences = []
    test_sequences = []

    groups = sorted(df["group"].unique())

    print("SEQUENCE SPLITTING")
    print("-" * 75)

    for group in groups:

        group_sequences = [
            item["sequence_id"]
            for item in sequences
            if item["group"] == group
        ]

        random.shuffle(group_sequences)

        n = len(group_sequences)

        n_train = round(n * TRAIN_RATIO)
        n_val = round(n * VAL_RATIO)

        # Make sure at least one sequence remains for test.
        if n_train + n_val >= n:
            n_val = max(1, n_val)
            n_train = n - n_val - 1

        n_test = n - n_train - n_val

        train_part = group_sequences[:n_train]
        val_part = group_sequences[n_train:n_train + n_val]
        test_part = group_sequences[n_train + n_val:]

        train_sequences.extend(train_part)
        val_sequences.extend(val_part)
        test_sequences.extend(test_part)

        print(
            f"{group:<30} | "
            f"Train: {len(train_part):2d} | "
            f"Val: {len(val_part):2d} | "
            f"Test: {len(test_part):2d}"
        )

    print()

    # ------------------------------------------------------------
    # Convert lists to sets for leakage checking.
    # ------------------------------------------------------------
    train_set = set(train_sequences)
    val_set = set(val_sequences)
    test_set = set(test_sequences)

    print("SEQUENCE LEAKAGE CHECK")
    print("-" * 75)

    train_val = train_set & val_set
    train_test = train_set & test_set
    val_test = val_set & test_set

    print(f"Train âˆ© Validation: {len(train_val)}")
    print(f"Train âˆ© Test:       {len(train_test)}")
    print(f"Validation âˆ© Test:  {len(val_test)}")

    if train_val or train_test or val_test:
        raise ValueError(
            "SEQUENCE LEAKAGE DETECTED. "
            "Aborting before writing output files."
        )

    print("PASS: No plant/pot sequence appears in multiple splits.")
    print()

    # ------------------------------------------------------------
    # Assign every observation to the split of its sequence.
    # ------------------------------------------------------------
    def assign_split(sequence_id):
        if sequence_id in train_set:
            return "train"

        if sequence_id in val_set:
            return "val"

        if sequence_id in test_set:
            return "test"

        raise ValueError(
            f"Sequence has no assigned split: {sequence_id}"
        )

    df["split"] = df["sequence_id"].apply(assign_split)

    # ------------------------------------------------------------
    # Verify every sequence has only one split.
    # ------------------------------------------------------------
    split_check = df.groupby("sequence_id")["split"].nunique()

    bad_split_sequences = split_check[split_check > 1]

    if len(bad_split_sequences) > 0:
        raise ValueError(
            "A sequence was assigned to multiple splits."
        )

    # ------------------------------------------------------------
    # Distribution summary
    # ------------------------------------------------------------
    print("SEQUENCE DISTRIBUTION")
    print("-" * 75)

    for split_name in ["train", "val", "test"]:

        split_df = df[df["split"] == split_name]

        print()
        print(split_name.upper())

        for group in groups:
            count = (
                split_df.loc[
                    split_df["group"] == group,
                    "sequence_id"
                ]
                .nunique()
            )

            print(f"{group:<30} : {count:2d} sequences")

        print(
            f"TOTAL SEQUENCES: "
            f"{split_df['sequence_id'].nunique()}"
        )

    print()

    # ------------------------------------------------------------
    # Observation/frame distribution
    # ------------------------------------------------------------
    print("OBSERVATION DISTRIBUTION")
    print("-" * 75)

    for split_name in ["train", "val", "test"]:

        split_df = df[df["split"] == split_name]

        print(
            f"{split_name:<5}: "
            f"{len(split_df)} observations"
        )

    print()

    # ------------------------------------------------------------
    # Temporal coverage
    # ------------------------------------------------------------
    print("TEMPORAL COVERAGE")
    print("-" * 75)

    for split_name in ["train", "val", "test"]:

        split_df = df[df["split"] == split_name]

        days = sorted(split_df["day"].dropna().unique())

        print(
            f"{split_name:<5}: "
            f"{len(days)} unique days | "
            f"{days}"
        )

    print()

    # ------------------------------------------------------------
    # Group/class distribution by observations
    # ------------------------------------------------------------
    print("GROUP OBSERVATION DISTRIBUTION")
    print("-" * 75)

    for split_name in ["train", "val", "test"]:

        split_df = df[df["split"] == split_name]

        print()
        print(split_name.upper())

        for group in groups:
            count = len(
                split_df[split_df["group"] == group]
            )

            print(f"{group:<30} : {count}")

    print()

    # ------------------------------------------------------------
    # Modality completeness check
    # ------------------------------------------------------------
    print("MODALITY COMPLETENESS")
    print("-" * 75)

    modality_columns = [
        "rgb_path",
        "vnir_path",
        "vnir_800nm_path",
        "vnir_1000nm_path",
    ]

    for split_name in ["train", "val", "test"]:

        split_df = df[df["split"] == split_name]

        incomplete = split_df[
            split_df[modality_columns].isna().any(axis=1)
        ]

        print(
            f"{split_name:<5}: "
            f"{len(incomplete)} incomplete observations"
        )

    # ------------------------------------------------------------
    # We only want fully multimodal observations.
    # The temporal index should already contain 10178 aligned rows.
    # ------------------------------------------------------------
    incomplete_rows = df[
        df[modality_columns].isna().any(axis=1)
    ]

    if len(incomplete_rows) > 0:
        print()
        print(
            f"WARNING: {len(incomplete_rows)} observations "
            f"have missing modalities."
        )
        print(
            "These observations will be excluded from the "
            "final split CSV files."
        )

    final_df = df[
        ~df[modality_columns].isna().any(axis=1)
    ].copy()

    # ------------------------------------------------------------
    # Save split files
    # ------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_files = {
        "train": os.path.join(
            OUTPUT_DIR,
            "tobrfv_temporal_train.csv"
        ),
        "val": os.path.join(
            OUTPUT_DIR,
            "tobrfv_temporal_val.csv"
        ),
        "test": os.path.join(
            OUTPUT_DIR,
            "tobrfv_temporal_test.csv"
        ),
    }

    print()
    print("OUTPUT FILES")
    print("-" * 75)

    for split_name, output_file in output_files.items():

        split_df = final_df[
            final_df["split"] == split_name
        ].copy()

        # Sort chronologically inside each sequence.
        split_df = split_df.sort_values(
            by=["sequence_id", "day", "angle"]
        )

        split_df.to_csv(
            output_file,
            index=False
        )

        print(
            f"{split_name:<5}: "
            f"{len(split_df):5d} observations -> "
            f"{output_file}"
        )

    # ------------------------------------------------------------
    # Final verification
    # ------------------------------------------------------------
    print()
    print("FINAL VERIFICATION")
    print("=" * 75)

    saved_train = pd.read_csv(output_files["train"])
    saved_val = pd.read_csv(output_files["val"])
    saved_test = pd.read_csv(output_files["test"])

    saved_train_set = set(saved_train["sequence_id"])
    saved_val_set = set(saved_val["sequence_id"])
    saved_test_set = set(saved_test["sequence_id"])

    print(
        "Saved train sequences:",
        len(saved_train_set)
    )
    print(
        "Saved validation sequences:",
        len(saved_val_set)
    )
    print(
        "Saved test sequences:",
        len(saved_test_set)
    )

    print()

    print(
        "Saved Train âˆ© Val:",
        len(saved_train_set & saved_val_set)
    )
    print(
        "Saved Train âˆ© Test:",
        len(saved_train_set & saved_test_set)
    )
    print(
        "Saved Val âˆ© Test:",
        len(saved_val_set & saved_test_set)
    )

    if (
        saved_train_set & saved_val_set
        or saved_train_set & saved_test_set
        or saved_val_set & saved_test_set
    ):
        raise ValueError(
            "FINAL VERIFICATION FAILED: sequence leakage."
        )

    print()
    print("PASS: No overlap of constructed group-plus-pot IDs. Biological/temporal independence remains unvalidated.")

    print()
    print("=" * 75)
    print("DONE")
    print("=" * 75)


if __name__ == "__main__":
    run_preparation(main, [os.path.join(OUTPUT_DIR, "tobrfv_temporal_" + split + ".csv") for split in ("train", "val", "test")], unvalidated=True)