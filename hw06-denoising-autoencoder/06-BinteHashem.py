"""Generate nine digits and cluster all 48,000 training encodings."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch


def fit_clusters(codes):
    best = None
    for seed in range(559, 569):
        centers, _ = kmeans2(codes, 10, iter=50, minit="++", seed=seed)
        squared = cdist(codes, centers, metric="sqeuclidean")
        assignments = squared.argmin(axis=1)
        inertia = squared[np.arange(len(codes)), assignments].sum()
        if best is None or inertia < best[0]:
            best = (inertia, assignments)
    return best


@torch.inference_mode()
def solve(encoder, decoder, training, output):
    encoder.eval()
    decoder.eval()
    rng = torch.Generator().manual_seed(559)
    codes, labels = [], []
    for images, digits in training:
        codes.append(encoder(images).numpy())
        labels.append(digits.numpy())
    codes, labels = np.concatenate(codes), np.concatenate(labels)
    assert codes.shape == (48000, 4) and labels.shape == (48000,)

    centers = torch.from_numpy(codes)[torch.randperm(len(codes), generator=rng)[:9]]
    scale = torch.from_numpy(codes.std(axis=0))
    latent = centers + 0.05 * scale * torch.randn((9, 4), generator=rng)
    generated = decoder(latent).squeeze(1).numpy()
    fig, axes = plt.subplots(3, 3, figsize=(5, 5))
    for image, ax in zip(generated, axes.flat):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "generated-digits.png", dpi=150)
    plt.close(fig)

    inertia, cluster = fit_clusters(codes)
    counts = np.zeros((10, 10), dtype=np.int64)
    np.add.at(counts, (cluster, labels), 1)
    rows, columns = linear_sum_assignment(-counts)
    correct = int(counts[rows, columns].sum())
    return dict(samples=len(labels), accuracy=correct / len(labels), correct=correct,
                inertia=float(inertia), confusion=counts.tolist(),
                mapping=dict(zip(map(int, rows), map(int, columns))))


def save_results(encoder, decoder, training, output, history=None):
    output.mkdir(exist_ok=True)
    metrics = solve(encoder.cpu(), decoder.cpu(), training, output)
    if history is not None:
        metrics["history"] = history
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f'Matched training accuracy: {metrics["accuracy"]:.2%}')


def main(output):
    from starter import Decoder, Encoder, make_loaders

    checkpoint = torch.load(output / "autoencoder.pt", map_location="cpu", weights_only=True)
    encoder, decoder = Encoder(4, 128), Decoder(4, 128)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    training, _ = make_loaders()
    save_results(encoder, decoder, training, output, checkpoint["history"])


if __name__ == "__main__":
    torch.set_num_threads(4)
    output = Path(__file__).resolve().parent / "results"
    if all(name in globals() for name in ("encoder", "decoder", "train_loader")):
        # When appended to the course starter, use the live models and split.
        save_results(globals()["encoder"], globals()["decoder"], globals()["train_loader"], output)
    else:
        main(output)
