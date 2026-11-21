"""Train a character LSTM on the provided 2,000 names."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ALPHABET = " abcdefghijklmnopqrstuvwxyz"


def encode(names):
    inputs, targets = [], []
    for name in names:
        ids = [ALPHABET.index(letter) for letter in name]
        if not 1 <= len(ids) <= 11 or 0 in ids:
            raise ValueError(f"Invalid name: {name!r}")
        padded = ids + [0] * (12 - len(ids))
        inputs.append(padded[:-1])
        targets.append(padded[1:])
    return (F.one_hot(torch.tensor(inputs), 27).float(),
            F.one_hot(torch.tensor(targets), 27).float())


class NameLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(27, 96, batch_first=True)
        self.output = nn.Linear(96, 27)

    def forward(self, inputs):
        hidden, _ = self.lstm(inputs)
        return self.output(hidden)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=180)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.set_num_threads(3)
    torch.manual_seed(559)
    names = (HERE / "data/names.txt").read_text().lower().splitlines()
    x, y = encode(names)
    assert x.shape == y.shape == (2000, 11, 27)
    target = y.argmax(dim=-1)
    model = NameLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0.0001)
    history = []
    for epoch in range(args.epochs):
        for indices in torch.randperm(len(x)).split(128):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x[indices]).transpose(1, 2), target[indices])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            mean = sum(
                F.cross_entropy(model(a).transpose(1, 2), b, reduction="sum").item()
                for a, b in zip(x.split(128), target.split(128))
            ) / target.numel()
        history.append(mean)
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: {mean:.5f}", flush=True)

    output = HERE / "results"
    output.mkdir(exist_ok=True)
    torch.jit.trace(model.eval(), x[:1]).save(str(HERE / "0702-BinteHashem.pt"))
    (output / "loss.json").write_text(json.dumps(history, indent=2) + "\n")
    plt.plot(range(1, len(history) + 1), history)
    plt.xlabel("Epoch")
    plt.ylabel("Mean next-character cross-entropy")
    plt.tight_layout()
    plt.savefig(output / "loss.png", dpi=150)


if __name__ == "__main__":
    main()
