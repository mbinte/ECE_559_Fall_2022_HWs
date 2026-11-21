"""Fit the HW04 curve with a 1×24×1 network and online backpropagation."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def predict(x, a, b, v, c):
    return np.tanh(x[:, None] * a + b) @ v + c


def main():
    rng = np.random.default_rng(559)
    x = rng.uniform(0, 1, 300)
    noise = rng.uniform(-0.1, 0.1, len(x))
    d = np.sin(20 * x) + 3 * x + noise
    a = rng.uniform(11, 20, 24)
    b = -a * np.linspace(0, 1, 24)
    v = rng.uniform(-0.1, 0.1, 24)
    c = 0.0
    history = [float(np.mean((d - predict(x, a, b, v, c)) ** 2))]
    best_loss, best = history[0], (a.copy(), b.copy(), v.copy(), c)
    total_epochs = 3500
    for epoch in range(total_epochs):
        eta = 0.015 * (0.03 ** (epoch / (total_epochs - 1)))
        for i in rng.permutation(len(x)):
            xi = x[i]
            hidden = np.tanh(a * xi + b)
            residual = c + hidden @ v - d[i]
            output_delta = 2 * residual
            hidden_delta = output_delta * v * (1 - hidden**2)
            a -= eta * hidden_delta * xi
            b -= eta * hidden_delta
            v -= eta * output_delta * hidden
            c -= eta * output_delta
        loss = float(np.mean((d - predict(x, a, b, v, c)) ** 2))
        history.append(loss)
        if loss < best_loss:
            best_loss, best = loss, (a.copy(), b.copy(), v.copy(), c)

    output = HERE / "results"
    output.mkdir(exist_ok=True)
    np.savez(output / "weights.npz", a=best[0], b=best[1], v=best[2], c=best[3])
    (output / "metrics.json").write_text(json.dumps(
        dict(mse=history, best_epoch=int(np.argmin(history)), best_mse=best_loss), indent=2) + "\n")
    fig, ax = plt.subplots()
    ax.semilogy(range(len(history)), history)
    ax.set(xlabel="Epoch", ylabel="Training MSE", title="Online backpropagation")
    fig.tight_layout()
    fig.savefig(output / "mse.png", dpi=150)
    plt.close(fig)
    grid = np.linspace(0, 1, 1000)
    fig, ax = plt.subplots()
    ax.scatter(x, d, s=9, alpha=0.65, label="Noisy samples")
    ax.plot(grid, predict(grid, *best), color="black", label="Neural network fit")
    ax.set(xlabel="x", ylabel="d", title="Fitted 1×24×1 tanh network")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "fit.png", dpi=150)
    plt.close(fig)
    print(f"Best training MSE: {best_loss:.5f}")


if __name__ == "__main__":
    main()
