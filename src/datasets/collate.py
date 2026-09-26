"""Preserved resize/tensor/padding collator. No inferred severity or risk targets."""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.utils.config import load_config


# ------------------------------------------------------------
# Transform for PIL images
# ------------------------------------------------------------

image_transform = transforms.Compose([
    transforms.Resize((load_config("preprocessing")["images"]["size"],) * 2),
    transforms.ToTensor(),
])


# ------------------------------------------------------------
# Convert one temporal sequence
# ------------------------------------------------------------

def transform_sequence(sequence):
    """
    Convert a list of PIL images into:

        [T, C, H, W]
    """

    tensors = []

    for image in sequence:
        tensors.append(image if torch.is_tensor(image) else image_transform(image))

    return torch.stack(tensors, dim=0)


# ------------------------------------------------------------
# Pad temporal sequence
# ------------------------------------------------------------

def pad_sequence(sequence, max_len):
    """
    Input:
        [T, C, H, W]

    Output:
        [max_len, C, H, W]
    """

    length = sequence.shape[0]

    if length == max_len:
        return sequence

    padding = torch.zeros(
        (
            max_len - length,
            *sequence.shape[1:],
        ),
        dtype=sequence.dtype,
    )

    return torch.cat(
        [sequence, padding],
        dim=0,
    )


# ------------------------------------------------------------
# Collate function
# ------------------------------------------------------------

def collate_fn(batch):

    batch_size = len(batch)

    # --------------------------------------------------------
    # Sequence lengths
    # --------------------------------------------------------

    lengths = [
        len(item["days"])
        for item in batch
    ]

    max_len = max(lengths)

    # --------------------------------------------------------
    # Transform each modality
    # --------------------------------------------------------

    rgb_sequences = [
        transform_sequence(item["rgb"])
        for item in batch
    ]

    vnir_sequences = [
        transform_sequence(item["vnir"])
        for item in batch
    ]

    vnir_800_sequences = [
        transform_sequence(item["vnir_800nm"])
        for item in batch
    ]

    vnir_1000_sequences = [
        transform_sequence(item["vnir_1000nm"])
        for item in batch
    ]

    # --------------------------------------------------------
    # Pad sequences
    # --------------------------------------------------------

    rgb = torch.stack(
        [
            pad_sequence(sequence, max_len)
            for sequence in rgb_sequences
        ],
        dim=0,
    )

    vnir = torch.stack(
        [
            pad_sequence(sequence, max_len)
            for sequence in vnir_sequences
        ],
        dim=0,
    )

    vnir_800nm = torch.stack(
        [
            pad_sequence(sequence, max_len)
            for sequence in vnir_800_sequences
        ],
        dim=0,
    )

    vnir_1000nm = torch.stack(
        [
            pad_sequence(sequence, max_len)
            for sequence in vnir_1000_sequences
        ],
        dim=0,
    )

    # --------------------------------------------------------
    # Temporal metadata
    # --------------------------------------------------------

    days = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
    )

    angles = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
    )

    mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.bool,
    )

    for i, item in enumerate(batch):

        length = len(item["days"])

        days[i, :length] = torch.as_tensor(
            item["days"],
            dtype=torch.long,
        )

        angles[i, :length] = torch.as_tensor(
            item["angles"],
            dtype=torch.long,
        )

        mask[i, :length] = True

    # --------------------------------------------------------
    # Return batch
    # --------------------------------------------------------

    return {
        "rgb": rgb,
        "vnir": vnir,
        "vnir_800nm": vnir_800nm,
        "vnir_1000nm": vnir_1000nm,
        "days": days,
        "angles": angles,
        "mask": mask,
        "lengths": torch.tensor(
            lengths,
            dtype=torch.long,
        ),
        "statuses": [item["status"] for item in batch],
        "groups": [item["group"] for item in batch],
        "pot_ids": [item["pot_id"] for item in batch],
        "sequence_ids": [
            item["sequence_id"]
            for item in batch
        ],
    }


