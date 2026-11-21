"""Print one shape prediction per image in the current directory."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ("Circle", "Square", "Octagon", "Heptagon", "Nonagon",
           "Star", "Hexagon", "Pentagon", "Triangle")
MODEL_NAME = "0602-BinteHashem.npz"
HERE = Path(__file__).resolve().parent
ANGLE_COUNT = 256


def read_mask(path):
    with Image.open(path) as image:
        if image.size != (200, 200):
            raise ValueError(f"{path}: expected a 200×200 image, got {image.size}")
        rgb = np.asarray(image.convert("RGB"))
    corners = rgb[[0, 0, -1, -1], [0, -1, 0, -1]]
    colors, counts = np.unique(corners, axis=0, return_counts=True)
    mask = np.any(rgb != colors[counts.argmax()], axis=2)
    if not mask.any():
        raise ValueError(f"{path}: no foreground pixels found")
    return mask


def pixel_features(mask):
    # Average each 10×10 tile so the baseline keeps coarse spatial information.
    return mask.reshape(20, 10, 20, 10).mean(axis=(1, 3)).ravel()


def radial_features(mask):
    rows, cols = np.nonzero(mask)
    y = rows - rows.mean()
    x = cols - cols.mean()
    angles = np.arctan2(y, x)
    bins = np.floor((angles + np.pi) * (ANGLE_COUNT / (2 * np.pi))).astype(int)
    radii = np.zeros(ANGLE_COUNT)
    np.maximum.at(radii, bins % ANGLE_COUNT, np.hypot(x, y))
    occupied = np.flatnonzero(radii)
    if not occupied.size:
        raise ValueError("Foreground must contain more than one pixel")
    radii = np.interp(np.arange(ANGLE_COUNT), occupied, radii[occupied],
                      period=ANGLE_COUNT)
    normalized = radii / radii.mean()
    # Rotation shifts the radial signal. Its Fourier magnitudes retain the
    # periodicity from corners while discarding the starting angle.
    spectrum = np.abs(np.fft.rfft(normalized)) / ANGLE_COUNT
    return np.r_[spectrum[1:33], normalized.std(), normalized.min(), normalized.max()]


def logits(x, w1, b1, w2, b2):
    return np.maximum(x @ w1 + b1, 0) @ w2 + b2


def main():
    default_model = HERE / MODEL_NAME
    if not default_model.exists():
        default_model = HERE / "results" / MODEL_NAME
    if not default_model.exists():
        default_model = HERE / "full-results" / MODEL_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=default_model)
    args = parser.parse_args()
    with np.load(args.model, allow_pickle=False) as model:
        classes = model["classes"]
        mean, scale = model["mean"], model["scale"]
        params = [model[key] for key in ("w1", "b1", "w2", "b2")]
    for path in sorted(Path.cwd().iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            features = (radial_features(read_mask(path)) - mean) / scale
            print(f"{path.name}: {classes[logits(features, *params).argmax()]}")


if __name__ == "__main__":
    main()
