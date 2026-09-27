"""Prototype day/angle/pot loader; biological identity is not yet validated."""
import os
from pathlib import Path
from src.utils.config import manifest_path, project_path, resolve_image_path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class TOBRFVTemporalDataset(Dataset):

    MODALITIES = [
        "rgb",
        "vnir",
        "vnir_800nm",
        "vnir_1000nm",
    ]

    PATH_COLUMNS = {
        "rgb": "rgb_path",
        "vnir": "vnir_path",
        "vnir_800nm": "vnir_800nm_path",
        "vnir_1000nm": "vnir_1000nm_path",
    }

    def __init__(
        self,
        csv_file,
        transform=None,
        max_sequence_length=None,
    ):
        csv_file = Path(csv_file)
        if not csv_file.is_absolute():
            csv_file = manifest_path(csv_file) if csv_file.parent == Path(".") else project_path(csv_file)
        if max_sequence_length is not None and max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        self.csv_file = csv_file
        self.transform = transform
        self.max_sequence_length = max_sequence_length

        if not os.path.exists(csv_file):
            raise FileNotFoundError(
                f"CSV file not found:\n{csv_file}"
            )

        self.df = pd.read_csv(csv_file)

        if self.df.empty:
            raise ValueError(
                f"CSV file is empty:\n{csv_file}"
            )

        required_columns = [
            "group",
            "status",
            "day",
            "angle",
            "pot_id",
            "sequence_id",
            "rgb_path",
            "vnir_path",
            "vnir_800nm_path",
            "vnir_1000nm_path",
        ]

        missing = [
            column
            for column in required_columns
            if column not in self.df.columns
        ]

        if missing:
            raise ValueError(
                "Missing required columns:\n"
                + "\n".join(missing)
            )

        self.df["day"] = pd.to_numeric(
            self.df["day"],
            errors="raise"
        )

        self.df["angle"] = pd.to_numeric(
            self.df["angle"],
            errors="raise"
        )

        path_columns = list(
            self.PATH_COLUMNS.values()
        )

        incomplete = self.df[
            self.df[path_columns].isna().any(axis=1)
        ]

        if not incomplete.empty:
            raise ValueError(
                f"Found {len(incomplete)} observations "
                "with missing modalities."
            )

        sequence_groups = (
            self.df.groupby("sequence_id")["group"]
            .nunique()
        )

        invalid_sequences = sequence_groups[
            sequence_groups > 1
        ]

        if not invalid_sequences.empty:
            raise ValueError(
                "Some sequence IDs belong to multiple groups."
            )

        self.df = self.df.sort_values(
            by=[
                "sequence_id",
                "day",
                "angle",
            ]
        ).reset_index(drop=True)

        self.sequence_ids = (
            self.df["sequence_id"]
            .drop_duplicates()
            .tolist()
        )

        self.sequence_data = {}

        for sequence_id in self.sequence_ids:

            sequence_df = self.df[
                self.df["sequence_id"] == sequence_id
            ].copy()

            sequence_df = sequence_df.sort_values(
                by=["day", "angle"]
            )

            self.sequence_data[sequence_id] = sequence_df

    def __len__(self):
        return len(self.sequence_ids)

    def _load_image(self, path):

        path = resolve_image_path(path)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Image not found:\n{path}"
            )

        with Image.open(path) as image:
            return image.copy()

    def _apply_transform(self, image):

        if self.transform is None:
            return image

        return self.transform(image)

    def __getitem__(self, index):

        sequence_id = self.sequence_ids[index]

        sequence_df = self.sequence_data[
            sequence_id
        ]

        if (
            self.max_sequence_length is not None
            and len(sequence_df) > self.max_sequence_length
        ):
            sequence_df = sequence_df.iloc[
                :self.max_sequence_length
            ]

        rgb_sequence = []
        vnir_sequence = []
        vnir_800_sequence = []
        vnir_1000_sequence = []

        days = []
        angles = []

        for _, row in sequence_df.iterrows():

            rgb = self._load_image(
                row["rgb_path"]
            )

            vnir = self._load_image(
                row["vnir_path"]
            )

            vnir_800 = self._load_image(
                row["vnir_800nm_path"]
            )

            vnir_1000 = self._load_image(
                row["vnir_1000nm_path"]
            )

            rgb = self._apply_transform(rgb)
            vnir = self._apply_transform(vnir)
            vnir_800 = self._apply_transform(vnir_800)
            vnir_1000 = self._apply_transform(vnir_1000)

            rgb_sequence.append(rgb)
            vnir_sequence.append(vnir)
            vnir_800_sequence.append(vnir_800)
            vnir_1000_sequence.append(vnir_1000)

            days.append(int(row["day"]))
            angles.append(int(row["angle"]))

        if (
            rgb_sequence
            and torch.is_tensor(rgb_sequence[0])
        ):
            rgb_sequence = torch.stack(
                rgb_sequence
            )

            vnir_sequence = torch.stack(
                vnir_sequence
            )

            vnir_800_sequence = torch.stack(
                vnir_800_sequence
            )

            vnir_1000_sequence = torch.stack(
                vnir_1000_sequence
            )

        return {
            "rgb": rgb_sequence,
            "vnir": vnir_sequence,
            "vnir_800nm": vnir_800_sequence,
            "vnir_1000nm": vnir_1000_sequence,
            "days": torch.tensor(
                days,
                dtype=torch.long
            ),
            "angles": torch.tensor(
                angles,
                dtype=torch.long
            ),
            "sequence_id": sequence_id,
            "group": sequence_df.iloc[0]["group"],
            "status": sequence_df.iloc[0]["status"],
            "pot_id": int(
                sequence_df.iloc[0]["pot_id"]
            ),
        }

    def get_sequence_lengths(self):

        return [
            len(
                self.sequence_data[sequence_id]
            )
            for sequence_id in self.sequence_ids
        ]

    def get_sequence_info(self, index):

        sequence_id = self.sequence_ids[index]

        sequence_df = self.sequence_data[
            sequence_id
        ]

        return {
            "sequence_id": sequence_id,
            "group": sequence_df.iloc[0]["group"],
            "status": sequence_df.iloc[0]["status"],
            "pot_id": int(
                sequence_df.iloc[0]["pot_id"]
            ),
            "num_observations": len(sequence_df),
            "days": sorted(
                sequence_df["day"].unique().tolist()
            ),
            "angles": sorted(
                sequence_df["angle"].unique().tolist()
            ),
        }


def create_dataset(
    split="train",
    transform=None,
    max_sequence_length=None,
):

    if split not in [
        "train",
        "val",
        "test",
    ]:
        raise ValueError(
            "split must be 'train', 'val', or 'test'"
        )

    csv_file = manifest_path(f"tobrfv_temporal_{split}.csv")

    return TOBRFVTemporalDataset(
        csv_file=csv_file,
        transform=transform,
        max_sequence_length=max_sequence_length,
    )