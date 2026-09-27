"""Load at most two observations from each of two historical sequences."""
from torch.utils.data import DataLoader, Subset
from src.datasets.temporal import create_dataset
from src.datasets.collate import collate_fn


def main():
    dataset = create_dataset(max_sequence_length=2)
    batch = next(iter(DataLoader(Subset(dataset, range(min(2, len(dataset)))), batch_size=2, collate_fn=collate_fn)))
    print({key: tuple(value.shape) for key, value in batch.items() if hasattr(value, "shape")})
    assert batch["mask"].sum().item() == batch["lengths"].sum().item()


if __name__ == "__main__":
    main()
