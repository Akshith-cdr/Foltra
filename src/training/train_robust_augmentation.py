"""Train the isolated Week 6 field-style augmentation candidate."""
import argparse

import torch

from src.training.train_baseline import run
from src.utils.config import load_config


FAIRNESS_KEYS = ("epochs", "batch_size", "learning_rate", "weight_decay")


def format_duration(seconds):
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def report_epoch_progress(row, epoch, total_epochs, elapsed):
    """Print one compact, Colab-readable summary after a completed epoch."""
    eta = elapsed / epoch * (total_epochs - epoch)
    print(
        f"Epoch {epoch}/{total_epochs} complete | "
        f"train loss {row['train_loss']:.6f} | "
        f"validation loss {row['validation_loss']:.6f} | "
        f"validation accuracy {100 * row['validation_accuracy']:.2f}% | "
        f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}",
        flush=True,
    )


def validate_fairness(config, baseline=None):
    """Fail if a scientific control differs from the Week 5 baseline."""
    baseline = baseline or load_config("baseline")
    if config["manifests"] != baseline["manifests"]:
        raise ValueError("Week 6 must use the exact Week 5 manifests")
    differences = {
        key: (baseline["training"][key], config["training"][key])
        for key in FAIRNESS_KEYS
        if config["training"][key] != baseline["training"][key]
    }
    if differences:
        raise ValueError(f"Week 6 fairness controls differ from Week 5: {differences}")
    if config.get("seed") != baseline.get("seed"):
        raise ValueError("Week 6 must use the Week 5 random seed")
    if config.get("augmentation", {}).get("profile") != "field_style_v1":
        raise ValueError("Week 6 requires augmentation profile field_style_v1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="robust_augmentation")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_fairness(config)
    if not args.smoke and not torch.cuda.is_available():
        raise RuntimeError("Full Week 6 training is GPU-only; run it in the prepared Colab workflow")
    run(config, smoke=args.smoke, epoch_reporter=report_epoch_progress)


if __name__ == "__main__":
    main()
