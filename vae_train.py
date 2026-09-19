import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        # encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        self.latent_dim = latent_dim

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        mu = self.fc21(h1)
        log_var = self.fc22(h1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, log_var = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var


def vae_loss(recon_x, x, input_dim, mu, log_var, datasets_name):
    # reconstruct loss
    if datasets_name == 'MNIST':
        BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
    else:
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KL_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return BCE + KL_divergence


def vae_train(parameters, x_unlab, datasets_name, device):
    if isinstance(x_unlab, np.ndarray):
        x_unlab = torch.tensor(x_unlab, dtype=torch.float32).to(device)

    # network parameters
    hidden_dim = parameters['hidden_dim']
    latent_dim = parameters['latent_dim']
    batch_size = parameters['batch_size']
    epochs = parameters['epochs']

    # other parameters
    n, input_dim = x_unlab.shape
    train_loader = DataLoader(x_unlab, batch_size=batch_size, shuffle=True)

    # init model
    model = VAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=0.001,
                           weight_decay=1e-4
                           )

    # training
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(data)
            loss = vae_loss(recon_batch, data, input_dim, mu, log_var, datasets_name)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            # if batch_idx % 100 == 0:
            #     print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
            #           f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item() / len(data):.6f}")

        print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}")

    return model