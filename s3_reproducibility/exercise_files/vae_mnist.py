"""Adapted from https://github.com/Jackson-Kang/PyTorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb.

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""

import os
import hydra
import logging
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import Decoder, Encoder, Model
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image

log = logging.getLogger(__name__)


# Model Hyperparameters
@hydra.main(
    version_base=None,
    config_path="./conf",
    config_name="config",
)
def main(cfg):
    # -------------------------
    # Reproducibility
    # -------------------------
    seed = cfg.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # -------------------------
    # Hyperparameters
    # -------------------------
    hp = cfg.hyperparameters
    batch_size = hp.batch_size
    x_dim = hp.x_dim
    hidden_dim = hp.hidden_dim
    latent_dim = hp.latent_dim
    epochs = hp.epochs
    lr = hp.learning_rate

    log.info("Batch size: %s, Learning rate: %s, Seed: %s", batch_size, lr, seed)

    # -------------------------
    # Device
    # -------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Data
    # -------------------------
    dataset_path = "~/datasets"
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST(dataset_path, transform=transform, train=True, download=True)
    test_dataset = MNIST(dataset_path, transform=transform, train=False, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------
    # Model
    # -------------------------
    encoder = Encoder(x_dim, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, hidden_dim, x_dim)
    model = Model(encoder, decoder).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr)

    # -------------------------
    # Training loop
    # -------------------------
    log.info("Start training VAE...")
    model.train()

    for epoch in range(epochs):
        overall_loss = 0.0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, x_dim).to(DEVICE)

            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            loss.backward()
            optimizer.step()

            overall_loss += loss.item()

        log.info(
            "Epoch %s complete. Avg loss: %s",
            epoch + 1,
            overall_loss / len(train_loader.dataset),
        )

    log.info("Training finished")

    # -------------------------
    # Save artifacts
    # -------------------------
    torch.save(model, "trained_model.pt")

    # -------------------------
    # Evaluation
    # -------------------------
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.view(batch_size, x_dim).to(DEVICE)
            x_hat, _, _ = model(x)
            break

    save_image(x.view(batch_size, 1, 28, 28), "orig_data.png")
    save_image(x_hat.view(batch_size, 1, 28, 28), "reconstructions.png")

    # -------------------------
    # Sampling
    # -------------------------
    with torch.no_grad():
        noise = torch.randn(batch_size, latent_dim).to(DEVICE)
        generated_images = decoder(noise)

    save_image(
        generated_images.view(batch_size, 1, 28, 28),
        "generated_sample.png",
    )
