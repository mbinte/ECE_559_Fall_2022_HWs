"""Run the six binary perceptron experiments from HW02."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def train(x, target, initial, rate):
    weights = initial.copy()
    history = [int(np.count_nonzero((x @ weights >= 0) != target))]
    while history[-1]:
        for row, label in zip(x, target):
            prediction = int(row @ weights >= 0)
            weights += rate * (int(label) - prediction) * row
        history.append(int(np.count_nonzero((x @ weights >= 0) != target)))
    return weights, history


def main():
    rng = np.random.default_rng(559)
    truth = rng.uniform([-0.25, -1, -1], [0.25, 1, 1])
    initial = rng.uniform(-1, 1, 3)
    output = HERE / "results"
    output.mkdir(exist_ok=True)
    runs = {}
    for n in (100, 1000):
        x = np.column_stack((np.ones(n), rng.uniform(-1, 1, (n, 2))))
        target = x @ truth >= 0
        fig, ax = plt.subplots()
        for label in (False, True):
            ax.scatter(*x[target == label, 1:].T, s=9, marker="o" if label else "x",
                       label=f"S{int(label)}")
        if abs(truth[2]) > abs(truth[1]):
            u = np.array([-1, 1])
            ax.plot(u, -(truth[0] + truth[1] * u) / truth[2], label="True boundary")
        else:
            v = np.array([-1, 1])
            ax.plot(-(truth[0] + truth[2] * v) / truth[1], v, label="True boundary")
        ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel="x₁", ylabel="x₂", title=f"n = {n}")
        ax.legend()
        fig.savefig(output / f"classes-{n}.png", dpi=150)
        plt.close(fig)
        runs[n] = {}
        fig, ax = plt.subplots()
        for rate in (1, 10, 0.1):
            weights, history = train(x, target, initial, rate)
            runs[n][rate] = dict(weights=weights.tolist(), errors=history)
            ax.plot(range(len(history)), history, label=f"η = {rate}")
        ax.set(xlabel="Epoch", ylabel="Misclassifications", title=f"Perceptron learning, n = {n}")
        ax.legend()
        fig.savefig(output / f"errors-{n}.png", dpi=150)
        plt.close(fig)
    (output / "metrics.json").write_text(json.dumps(
        dict(truth=truth.tolist(), initial=initial.tolist(), runs=runs), indent=2) + "\n")


if __name__ == "__main__":
    main()
