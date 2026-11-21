"""Prepare the HW5 splits, train two MLPs, and export the radial model."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "hw5_inference", HERE / "0603-BinteHashem.py"
)
inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)


def prepare(data, output, subset):
    groups = {label: [] for label in inference.CLASSES}
    for path in sorted(data.glob("*.png")):
        label = path.name.split("_", 1)[0]
        if label in groups:
            groups[label].append(path)

    counts = {label: len(paths) for label, paths in groups.items()}
    if not all(n == 10000 for n in counts.values()) and not subset:
        raise SystemExit(f"Expected 10,000 images per class. Found {counts}. "
                         "Use --allow-subset only for a partial-data experiment.")
    if min(counts.values()) < 2:
        raise SystemExit(f"Each class needs at least two images. Found {counts}.")

    splits = {"training": [], "testing": []}
    signature = hashlib.sha256()
    for label, paths in groups.items():
        cut = len(paths) * 4 // 5
        for index, path in enumerate(paths):
            stat = path.stat()
            signature.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
            split = "training" if index < cut else "testing"
            splits[split].append((path, label))
    cache_key = f"features-v1:{signature.hexdigest()}"
    datasets = {}
    for split, entries in splits.items():
        target = output / f"{split}.npz"
        if target.exists():
            with np.load(target, allow_pickle=False) as saved:
                if saved["cache_key"].item() == cache_key:
                    datasets[split] = {key: saved[key] for key in saved.files}
                    continue

        radial, pixels = [], []
        for path, _ in entries:
            mask = inference.read_mask(path)
            radial.append(inference.radial_features(mask))
            pixels.append(inference.pixel_features(mask))
        datasets[split] = dict(
            radial=np.asarray(radial, dtype=np.float32),
            pixels=np.asarray(pixels, dtype=np.float32),
            labels=np.array([inference.CLASSES.index(label) for _, label in entries]),
            files=np.array([path.name for path, _ in entries]),
            cache_key=np.array(cache_key),
        )
        np.savez_compressed(target, **datasets[split])
    return datasets, counts


def measure(inputs, labels, params):
    logits = inference.logits(inputs, *params)
    shifted = logits - logits.max(axis=1, keepdims=True)
    loss = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]
    return float(loss.mean()), float((logits.argmax(axis=1) == labels).mean())


def train(training, testing, field, epochs):
    rng = np.random.default_rng(559)
    mean = training[field].mean(axis=0)
    scale = training[field].std(axis=0).clip(min=1e-5)
    x = (training[field] - mean) / scale
    test_x = (testing[field] - mean) / scale
    y, test_y = training["labels"], testing["labels"]
    width = 64
    w1 = rng.normal(0, np.sqrt(2 / x.shape[1]), (x.shape[1], width)).astype(np.float32)
    b1 = np.zeros(width, dtype=np.float32)
    w2 = rng.normal(0, np.sqrt(2 / width), (width, len(inference.CLASSES))).astype(np.float32)
    b2 = np.zeros(len(inference.CLASSES), dtype=np.float32)
    params = (w1, b1, w2, b2)
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    history = []
    step = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 256):
            indices = order[start:start + 256]
            batch = x[indices]
            hidden = np.maximum(batch @ w1 + b1, 0)
            scores = hidden @ w2 + b2
            probs = np.exp(scores - scores.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            probs[np.arange(len(indices)), y[indices]] -= 1
            probs /= len(indices)
            delta = (probs @ w2.T) * (hidden > 0)
            gradients = (batch.T @ delta, delta.sum(axis=0),
                         hidden.T @ probs, probs.sum(axis=0))
            step += 1
            for param, grad, m, v in zip(params, gradients, first, second):
                m *= 0.9
                m += 0.1 * grad
                v *= 0.999
                v += 0.001 * grad**2
                param -= 0.001 * (m / (1 - 0.9**step)) / (
                    np.sqrt(v / (1 - 0.999**step)) + 1e-8
                )
        train_loss, train_acc = measure(x, y, params)
        test_loss, test_acc = measure(test_x, test_y, params)
        history.append(dict(epoch=epoch, train_loss=train_loss, test_loss=test_loss,
                            train_accuracy=train_acc, test_accuracy=test_acc))
        print(f"{field:6s} {epoch:2d} train {train_loss:.4f} / {train_acc:.2%}"
              f"  test {test_loss:.4f} / {test_acc:.2%}", flush=True)
    return history, dict(mean=mean, scale=scale, w1=w1, b1=b1, w2=w2, b2=b2)


def plot(history, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric in ("loss", "accuracy"):
        fig, ax = plt.subplots(figsize=(7, 4))
        for split in ("train", "test"):
            values = [row[f"{split}_{metric}"] for row in history]
            if metric == "accuracy":
                values = [100 * value for value in values]
            ax.plot([row["epoch"] for row in history], values, label=split.title())
        ax.set(xlabel="Epoch", ylabel="Accuracy (%)" if metric == "accuracy" else
               "Mean cross-entropy", title=f"Radial MLP {metric}")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{metric}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "data" / "output")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    datasets, counts = prepare(args.data, args.output, args.allow_subset)
    histories = {}
    for field in ("pixels", "radial"):
        histories[field], model = train(datasets["training"], datasets["testing"],
                                        field, args.epochs)
    target = args.output / inference.MODEL_NAME
    np.savez_compressed(target, classes=np.array(inference.CLASSES), **model)
    if target.stat().st_size > 50_000_000:
        raise RuntimeError(f"Checkpoint exceeds 50 MB: {target}")
    split_counts = {
        split: dict(zip(inference.CLASSES, np.bincount(data["labels"], minlength=9).tolist()))
        for split, data in datasets.items()
    }
    (args.output / "metrics.json").write_text(json.dumps(
        dict(subset=any(n != 10000 for n in counts.values()), counts=split_counts,
             histories=histories, checkpoint_bytes=target.stat().st_size), indent=2
    ) + "\n")
    plot(histories["radial"], args.output)


if __name__ == "__main__":
    main()
