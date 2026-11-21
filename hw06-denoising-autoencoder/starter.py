"""Course denoising architecture with a reproducible MNIST training entry point."""
# Original architecture: https://github.com/eugeniaring/Medium-Articles/blob/main/Pytorch/denAE.ipynb

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import MNIST

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "datasets/mnist"


class Encoder(nn.Module):

    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            #nn.BatchNorm2d(8),
            nn.ReLU(True),
            # Second convolutional layer
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # Third convolutional layer
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            #nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        ### Flatten layer
        #self.flatten = torch.flatten(start_dim=1)

        ### Linear section
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = torch.flatten(x, start_dim=1)
        # # Apply linear layers
        x = self.encoder_lin(x)
        return x





class Decoder(nn.Module):

    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()

        ### Linear section
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )


        ### Convolutional section
        self.decoder_conv = nn.Sequential(
            # First transposed convolution
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # Second transposed convolution
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            # Third transposed convolution
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Unflatten
        x = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))(x)
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        # Apply a sigmoid to force the output to be between 0 and 1 (valid pixel values)
        x = torch.sigmoid(x)
        return x


def make_loaders():
    mnist = MNIST(DATA, train=True, download=False)
    dataset = TensorDataset(mnist.data.unsqueeze(1).float() / 255, mnist.targets)
    split_rng = torch.Generator().manual_seed(559)
    training, validation = random_split(dataset, [48000, 12000], generator=split_rng)
    return DataLoader(training, batch_size=256), DataLoader(validation, batch_size=256)


def add_noise(images):
    return (images + 0.3 * torch.randn_like(images)).clamp(0, 1)


def run_epoch(encoder, decoder, loader, device, optimizer=None):
    training = optimizer is not None
    encoder.train(training)
    decoder.train(training)
    total = 0.0
    count = 0
    with torch.set_grad_enabled(training):
        for images, _ in loader:
            images = images.to(device, memory_format=torch.channels_last)
            if training:
                optimizer.zero_grad()
            loss = nn.functional.mse_loss(decoder(encoder(add_noise(images))), images)
            if training:
                loss.backward()
                optimizer.step()
            total += loss.item() * len(images)
            count += len(images)
    return total / count


def train(epochs):
    torch.set_num_threads(4)
    torch.manual_seed(559)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training, validation = make_loaders()
    encoder = Encoder(4, 128).to(device, memory_format=torch.channels_last)
    decoder = Decoder(4, 128).to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=0.001)
    history = []
    output = HERE / "results"
    output.mkdir(exist_ok=True)
    for epoch in range(1, epochs + 1):
        training_loss = run_epoch(encoder, decoder, training, device, optimizer)
        validation_loss = run_epoch(encoder, decoder, validation, device)
        history.append([training_loss, validation_loss])
        print(f"{epoch:2d}: training {training_loss:.5f}, validation {validation_loss:.5f}", flush=True)
        if epoch == epochs or epoch % 5 == 0:
            torch.save(dict(encoder=encoder.state_dict(), decoder=decoder.state_dict(),
                            history=history, split_seed=559), output / "autoencoder.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    train(args.epochs)
