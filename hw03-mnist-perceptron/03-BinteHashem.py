"""Run the online multicategory PTA on the local MNIST data."""
import ctypes
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
MNIST = HERE.parent / "datasets/mnist/MNIST/raw"


def data(prefix):
    images = np.fromfile(MNIST / f"{prefix}-images-idx3-ubyte", dtype=np.uint8,
                         offset=16).reshape(-1, 784).astype(np.float32) / 255
    labels = np.fromfile(MNIST / f"{prefix}-labels-idx1-ubyte", dtype=np.uint8, offset=8)
    return images, labels


def load_kernel(directory):
    target = Path(directory) / "perceptron.so"
    subprocess.run(["cc", "-O3", "-ffast-math", "-shared", "-fPIC",
                    str(HERE / "perceptron.c"), "-o", str(target)], check=True)
    library = ctypes.CDLL(str(target))
    floats = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    labels = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
    library.online_epoch.argtypes = [floats, floats, labels, ctypes.c_size_t, ctypes.c_float]
    library.online_epoch.restype = None
    library.errors.argtypes = [floats, floats, labels, ctypes.c_size_t]
    library.errors.restype = ctypes.c_size_t
    return library


def experiment(kernel, training, testing, n, eta, epsilon, seed, limit):
    x, y = training
    test_x, test_y = testing
    weights = np.random.default_rng(seed).uniform(-0.01, 0.01, (10, 784)).astype(np.float32)
    history = []
    for epoch in range(limit + 1):
        wrong = kernel.errors(weights, x, y, n)
        history.append(wrong)
        if epoch < limit:
            # The prompt measures error before the pass and checks it after the pass.
            kernel.online_epoch(weights, x, y, n, eta)
        if wrong / n <= epsilon:
            break
    test_wrong = kernel.errors(weights, test_x, test_y, len(test_y))
    return dict(n=n, eta=eta, epsilon=epsilon, seed=seed, history=history,
                reached=bool(history[-1] / n <= epsilon), test_wrong=test_wrong,
                test_size=len(test_y))


def plot(runs, keys, target):
    fig, ax = plt.subplots()
    for key in keys:
        history = runs[key]["history"]
        ax.plot(range(len(history)), history, label=key)
    ax.set(xlabel="Epoch", ylabel="Training misclassifications")
    ax.legend()
    fig.savefig(target, dpi=150)
    plt.close(fig)


def main():
    output = HERE / "results"
    output.mkdir(exist_ok=True)
    training, testing = data("train"), data("t10k")
    with TemporaryDirectory() as directory:
        kernel = load_kernel(directory)
        runs = {}
        for key, n, limit in (("n=50", 50, 1500), ("n=1000", 1000, 2500),
                              ("n=60000", 60000, 70)):
            runs[key] = experiment(kernel, training, testing, n, 1, 0, 559, limit)
            run = runs[key]
            print(key, len(run["history"]) - 1, run["history"][-1], run["test_wrong"], flush=True)
        best = min(runs["n=60000"]["history"]) / 60000
        epsilon = min(1.0, (np.ceil(100 * best) + 1) / 100)
        for seed in (560, 561, 562):
            key = f"seed {seed}"
            runs[key] = experiment(kernel, training, testing, 60000, 1, epsilon, seed, 100)
            print(key, runs[key]["reached"], epsilon, runs[key]["test_wrong"], flush=True)
    for i, keys in enumerate((["n=50"], ["n=1000"], ["n=60000"],
                               ["seed 560", "seed 561", "seed 562"]), 1):
        plot(runs, keys, output / f"errors-{i}.png")
    (output / "metrics.json").write_text(json.dumps(runs, indent=2) + "\n")


if __name__ == "__main__":
    main()
