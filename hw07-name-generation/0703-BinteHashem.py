"""Generate twenty names from an initial letter."""
import argparse
from pathlib import Path
import string

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ALPHABET = " " + string.ascii_lowercase


@torch.inference_mode()
def generate(model, initial, temperature, rng):
    prefix = [ALPHABET.index(initial)]
    for _ in range(19):
        inputs = F.one_hot(torch.tensor([prefix]), 27).float()
        logits = model(inputs)[0, -1]
        probabilities = (logits / temperature).softmax(dim=0)
        next_id = torch.multinomial(probabilities, 1, generator=rng).item()
        if next_id == 0:
            break
        prefix.append(next_id)
    return "".join(ALPHABET[index] for index in prefix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("initial", type=str.lower, choices=string.ascii_lowercase)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float, default=0.85)
    args = parser.parse_args()
    if not 0 < args.temperature < float("inf"):
        parser.error("--temperature must be finite and positive")
    torch.set_num_threads(1)
    model = torch.jit.load(str(HERE / "0702-BinteHashem.pt"), map_location="cpu").eval()
    rng = torch.Generator()
    if args.seed is None:
        rng.seed()
    else:
        rng.manual_seed(args.seed)
    for _ in range(20):
        print(generate(model, args.initial, args.temperature, rng))


if __name__ == "__main__":
    main()
