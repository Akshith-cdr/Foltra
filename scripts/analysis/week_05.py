"""Analyze saved classification artifacts without loading images or model weights."""
import argparse
import csv
import hashlib
import json
import platform
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.config import project_path, ensure_output_path, manifest_path
from src.utils.manifests import load_manifest


DEFAULT_RUN = "experiments/week_05_baseline/outputs/train_20260916T171709Z_f582a465"
DEFAULT_OUTPUT = "experiments/week_05_baseline/analysis"
INPUTS = ("history.json", "run.json", "test/metrics.json", "test/confusion_matrix.csv")


def hashes(root):
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in INPUTS}


def table(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def divide(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def analyze(run_path, output):
    before = hashes(run_path)
    history = json.loads((run_path / "history.json").read_text())
    run = json.loads((run_path / "run.json").read_text())
    metrics = json.loads((run_path / "test/metrics.json").read_text())
    mapping = run["class_to_idx"]
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("Class indices must be contiguous")
    names = sorted(mapping, key=mapping.get)
    with (run_path / "test/confusion_matrix.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if rows[0][1:] != names or [row[0] for row in rows[1:]] != names:
        raise ValueError("CSV labels/order do not match the run class mapping")
    matrix = np.array([[int(value) for value in row[1:]] for row in rows[1:]], dtype=np.int64)
    if matrix.shape != (len(names), len(names)) or np.any(matrix < 0):
        raise ValueError("Invalid confusion matrix shape/counts")
    if not np.array_equal(matrix, np.array(metrics["confusion_matrix"])):
        raise ValueError("JSON and CSV confusion matrices disagree")
    support, predicted, tp = matrix.sum(axis=1), matrix.sum(axis=0), matrix.diagonal()
    precision, recall = divide(tp, predicted), divide(tp, support)
    f1 = divide(2 * tp, support + predicted)
    total, correct = int(matrix.sum()), int(tp.sum())
    accuracy, macro_f1 = correct / total, float(f1.mean())
    if total != metrics["samples"] or total != run["samples"]["test"]:
        raise ValueError("Test sample counts disagree")
    if not np.isclose(accuracy, metrics["accuracy"], atol=1e-12, rtol=0) or not np.isclose(macro_f1, metrics["macro_f1"], atol=1e-12, rtol=0):
        raise ValueError("Recomputed accuracy/F1 differ from saved results")
    manifest_checks = {}
    for split, name in run["config"]["manifests"].items():
        digest = hashlib.sha256(manifest_path(name).read_bytes()).hexdigest()
        if digest != run["manifest_sha256"][split]:
            raise ValueError(f"Current {split} manifest differs from training")
        manifest_checks[split] = digest
    counts = Counter(row["label"] for row in load_manifest(run["config"]["manifests"]["test"], required=("label",)))
    if counts != Counter({name: int(support[i]) for i, name in enumerate(names)}):
        raise ValueError("Test manifest label supports differ from confusion matrix")
    if run["smoke"] or run["status"] != "completed":
        raise ValueError("Expected a completed full run")
    best = min(history, key=lambda row: row["validation_loss"])
    if best["epoch"] != run["best_epoch"]:
        raise ValueError("Best epoch does not match minimum validation loss")
    highest_accuracy = max(history, key=lambda row: row["validation_accuracy"])
    output.mkdir(parents=True, exist_ok=True)
    table(output / "class_metrics.csv", ["class_index", "class", "precision", "recall", "f1", "support", "predicted_count", "true_positive", "false_positive", "false_negative"],
          [(i, name, precision[i], recall[i], f1[i], support[i], predicted[i], tp[i], predicted[i]-tp[i], support[i]-tp[i]) for i, name in enumerate(names)])
    confusions = sorted([(int(matrix[i, j]), i, j) for i in range(len(names)) for j in range(len(names)) if i != j and matrix[i, j]],
                        key=lambda item: (-item[0], names[item[1]], names[item[2]]))
    errors = total - correct
    table(output / "class_confusions.csv", ["true_class", "predicted_class", "count", "true_class_support", "fraction_of_true_class", "fraction_of_all_errors"],
          [(names[i], names[j], count, support[i], count/support[i], count/errors) for count, i, j in confusions])
    epochs = [row["epoch"] for row in history]
    plt.rcParams.update({"font.size": 11, "svg.hashsalt": "foltra-week05"})
    for filename, title, series, ylabel in [
        ("loss_curves", "PlantVillage Color: training and validation loss", [("train_loss", "Training"), ("validation_loss", "Validation")], "Cross-entropy loss"),
        ("validation_accuracy", "PlantVillage Color: validation accuracy", [("validation_accuracy", "Validation")], "Accuracy (%)")
    ]:
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        for key, label in series:
            values = [row[key] * (100 if key == "validation_accuracy" else 1) for row in history]
            ax.plot(epochs, values, marker="o", label=label)
        ax.axvline(best["epoch"], color="#666666", linestyle="--", label=f"Selected checkpoint: epoch {best['epoch']}")
        ax.set(xlabel="Epoch", ylabel=ylabel, title=title, xticks=epochs)
        ax.grid(alpha=0.25)
        ax.legend()
        for extension in ("png", "svg"):
            fig.savefig(output / f"{filename}.{extension}", dpi=180, metadata={"Software": "FOLTRA"} if extension == "png" else {"Date": None})
        plt.close(fig)
    # Counts and source-class error rates together make the ranking interpretable.
    top = confusions[:10]
    fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")
    labels = [f"{names[i]}\n  → {names[j]}" for _, i, j in top]
    ax.barh(range(len(top)), [count for count, _, _ in top], color="#3477a8")
    ax.set_yticks(range(len(top)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel="Misclassified test images", title="Largest directed class confusions (ranked by count)", xlim=(0, max(count for count, _, _ in top) * 1.45))
    for k, (count, i, _) in enumerate(top):
        ax.text(count + 0.2, k, f"{count} ({100*count/support[i]:.1f}% of true class)", va="center", fontsize=9)
    for extension in ("png", "svg"):
        fig.savefig(output / f"top_confusions.{extension}", dpi=180, metadata={"Software": "FOLTRA"} if extension == "png" else {"Date": None})
    plt.close(fig)
    weakest = sorted(range(len(names)), key=lambda i: (f1[i], names[i]))[:8]
    weak_table = markdown_table(["Class", "Precision", "Recall", "F1", "Support"],
                               [(names[i], f"{precision[i]:.4f}", f"{recall[i]:.4f}", f"{f1[i]:.4f}", str(support[i])) for i in weakest])
    confusion_table = markdown_table(["True class → predicted class", "Count", "% of true class"],
                                    [(f"{names[i]} → {names[j]}", str(count), f"{100*count/support[i]:.2f}%") for count, i, j in top])
    first, last = history[0], history[-1]
    within_crop = sum(count for count, i, j in confusions if names[i].split("___")[0] == names[j].split("___")[0])
    report = f"""# Week 5 baseline analysis

Source: `{run_path.name}`. Completed ImageNet-pretrained ResNet-18 fine-tuning,
38 PlantVillage Color classes, seed {run['config']['seed']}, {len(history)} epochs.
Splits: {run['samples']['train']:,} train / {run['samples']['val']:,} validation / {total:,} test.
Batch size {run['config']['training']['batch_size']}; AdamW learning rate {run['config']['training']['learning_rate']},
weight decay {run['config']['training']['weight_decay']}; 224-pixel RGB input.

## Training behavior and checkpoint selection

Training loss fell from {first['train_loss']:.5f} to {last['train_loss']:.5f}.
Validation loss fell overall from {first['validation_loss']:.5f} to {last['validation_loss']:.5f},
with a temporary spike at epoch 3 (0.30816) and another rise at epoch 9 (0.06958).
Validation accuracy increased from {100*first['validation_accuracy']:.2f}% to {100*last['validation_accuracy']:.2f}%.
Late validation improvements level off while training loss keeps declining; the short,
fluctuating series does not establish sustained severe overfitting. Training augmentation
and train/evaluation modes differ, so training and validation losses are not directly equivalent.

**Selected checkpoint: epoch {best['epoch']}**, with the lowest validation loss
**{best['validation_loss']:.5f}** and validation accuracy **{100*best['validation_accuracy']:.2f}%**.
The highest validation accuracy was epoch {highest_accuracy['epoch']} ({100*highest_accuracy['validation_accuracy']:.2f}%),
but selection used validation loss. The saved test results refer to the selected checkpoint.
No checkpoint was loaded or model evaluation repeated in this analysis.

![Loss curves](loss_curves.png)
![Validation accuracy](validation_accuracy.png)

## Test results

- Accuracy: **{100*accuracy:.2f}%** ({correct:,}/{total:,} correct; {errors} errors).
- Macro F1: **{macro_f1:.6f}**; support-weighted F1: **{np.average(f1, weights=support):.6f}**.
- Saved test cross-entropy loss: **{metrics['loss']:.6f}**.

Accuracy and macro F1 were independently recomputed from the count matrix and match
the saved metrics. Loss is quoted from metrics.json; counts cannot reconstruct it.

## Class-level observations

The complete 38-class table is [class_metrics.csv](class_metrics.csv), in saved class-index order.
Precision = TP/predicted count; recall = TP/true support; F1 = 2TP/(true support + predicted count).
Undefined ratios use zero, matching evaluation. Values below are fractions, not percentages.
The eight lowest-F1 classes are:

{weak_table}

Corn Cercospora/Gray Leaf Spot has perfect recall but reduced precision because other
classes are predicted as it; this differs from tomato Early Blight, whose main weakness
is missed true cases (recall 81.33%). Support ranges from {int(support.min())} to {int(support.max())} images.
Small classes such as healthy potato (23 images) have less stable estimates; high
aggregate accuracy should not obscure class imbalance. {int(np.sum(f1 == 1))} classes have F1 = 1 on this test split,
which does not imply perfect population performance.

## Confusion-matrix observations

Rows represent true labels; columns represent predictions. Ranked by off-diagonal count:

{confusion_table}

The largest pair is corn Northern Leaf Blight → Cercospora/Gray Leaf Spot:
20 images, 13.61% of that true class. Tomato Early Blight → Late Blight follows with
15 images, 10.00% of true Early Blight. Together these account for {100*35/errors:.2f}% of all errors.
{within_crop}/{errors} errors are within the same crop (crop inferred from the label prefix).
These counts locate weaknesses but do not prove visual similarity, annotation error,
or any other cause; reviewing individual images would be needed.

![Top confusions](top_confusions.png)

All nonzero directed confusions and both count-normalized rates are in
[class_confusions.csv](class_confusions.csv). Count ranking favors larger classes;
the accompanying source-class rates and supports provide context.

## Limitations

- PlantVillage test performance **does not establish field-image generalization**.
  No independent field dataset, lighting/background shift, or deployment test was evaluated.
- This is one seed and one fixed split; no repeated-run uncertainty estimates are available.
- Split hashes and class supports are verified, but this analysis does not audit duplicate
  images, plant-level independence, or possible split leakage.
- Aggregate artifacts support per-class metrics, but not calibration, ROC/PR curves,
  confidence analysis, or identifying particular misclassified image files.
- No raw images or model weights were read, no retraining occurred, and run outputs were unchanged.

## Reproduction

From the repository root, with the project dependencies and Matplotlib installed:

```powershell
python -B -m scripts.analysis.week_05
```

Optional arguments: `--run <repository-relative run directory>` and `--output <analysis directory>`.
The script regenerates its named files; keep personal notes elsewhere. PNG and SVG
plots, CSV tables, and this Markdown report are generated from the four saved inputs.
[provenance.json](provenance.json) records input/script hashes, checked manifest hashes,
and library versions. Original input hashes are checked again after generation.
"""
    (output / "analysis.md").write_text(report, encoding="utf-8")
    if hashes(run_path) != before:
        raise RuntimeError("Original inputs changed during analysis")
    provenance = {"run": run_path.name, "input_sha256": before, "manifest_sha256": manifest_checks,
                  "script_sha256": hashlib.sha256(project_path("scripts/analysis/week_05.py").read_bytes()).hexdigest(),
                  "versions": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
                  "checks": ["CSV/JSON matrices identical", "class order verified", "sample counts and class supports verified",
                             "accuracy and macro F1 reproduced", "manifest hashes match run", "best epoch matches minimum validation loss", "input hashes unchanged"]}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Analysis saved to {output}; {correct}/{total} correct, {errors} errors, macro F1 {macro_f1:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_path = project_path(args.run).resolve()
    output = ensure_output_path(args.output)
    if output == run_path or run_path in output.parents or output in run_path.parents:
        raise ValueError("Analysis directory must be separate from original run outputs")
    analyze(run_path, output)


if __name__ == "__main__":
    main()
