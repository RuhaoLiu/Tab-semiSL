import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from vae_train import VAE
from augment_methods import vae_generate_data, gaussian_sample


class TsSelfModel(nn.Module):
    def __init__(self, dim, proj_dim, dropout=0.5):
        super(TsSelfModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.mask_decoder = nn.Sequential(
            nn.Linear(proj_dim, dim),
            nn.Sigmoid()
        )
        self.feature_decoder = nn.Sequential(
            nn.Linear(proj_dim, dim),
            nn.Sigmoid()
        )

        # projection head
        self.projection_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, proj_dim),
        )

    def projection_encoder(self, x):
        h = self.encoder(x)
        g = self.projection_head(h)
        return g

    def forward(self, x):
        g = self.projection_encoder(x)
        mask_pred = self.mask_decoder(g)
        feature_pred = self.feature_decoder(g)
        return mask_pred, feature_pred, g


def info_nce_loss(z1, z2, temperature=0.5):
    """
    InfoNCE loss for contrastive learning.

    Args:
        - z1, z2: Two augmented views of the same batch. Shape: [batch_size, proj_dim]
        - temperature: Temperature scalar

    Returns:
        - Contrastive loss (scalar)
    """
    batch_size = z1.size(0)

    # Normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)

    # Similarity matrix
    sim = torch.mm(z, z.T) / temperature

    # Create labels: positives are (i, i + N) and (i + N, i)
    labels = torch.arange(batch_size).to(z1.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    # Mask out self-comparisons
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim = sim.masked_fill(mask, -1e9)  # Set diagonal to large negative value

    # Compute loss
    loss = F.cross_entropy(sim, labels)
    return loss


def con_mask(vae_file, x_unlab, p_m, alpha, beta, parameters, device):
    """
    Self auto-encoder model trained by contrastive learning and masked reconstruction

    Args:
      - vae_file: vae file name
      - x_unlab: numpy array or torch tensor，unlabeled features，shape (N, dim)
      - p_m: corruption probability
      - alpha: control the mask loss
      - beta: control the contrastive learning
      - parameters: model's parameters
      - device: cpu or cuda

    Returns:
      - encoder: trained self encoder model
    """

    if isinstance(x_unlab, np.ndarray):
        x_unlab = torch.tensor(x_unlab, dtype=torch.float32).to(device)

    epochs = parameters['epochs']
    batch_size = parameters['batch_size']
    vae_latent_dim = parameters['vae_latent_dim']
    vae_hidden_dim = parameters['vae_hidden_dim']
    N, dim = x_unlab.shape
    pro_dim = 64

    # load vae model
    VAE_model = VAE(input_dim=x_unlab.shape[1], hidden_dim=vae_hidden_dim, latent_dim=vae_latent_dim).to(device)
    VAE_model.load_state_dict(torch.load(vae_file, map_location=device))
    VAE_model.eval()

    # data perturbation
    m_label, x_u_perturbed = vae_generate_data(VAE_model, x_unlab, p_m)
    # m_label, x_u_perturbed = gaussian_sample(x_unlab, p_m)
    x_u_weak = x_unlab

    # init self_model of Tab-semiSL
    Ts_self_model = TsSelfModel(dim, pro_dim).to(device)
    optimizer = optim.RMSprop(
        Ts_self_model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
    )

    bce_loss = nn.BCELoss().to(device)
    mse_loss = nn.MSELoss().to(device)
    m_label = torch.tensor(m_label, dtype=torch.float32).to(device)
    x_u_perturbed = torch.tensor(x_u_perturbed, dtype=torch.float32).to(device)
    x_u_weak = torch.tensor(x_u_weak, dtype=torch.float32).to(device)
    dataset = TensorDataset(m_label, x_u_perturbed, x_u_weak)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # training self_model of Tab-semiSL
    Ts_self_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_m_label, batch_x_u_perturbed, batch_x_u_weak in dataloader:
            optimizer.zero_grad()

            # masked reconstruction
            mask_pred, feature_pred, _ = Ts_self_model(batch_x_u_perturbed)
            strong_latent_z = Ts_self_model.projection_encoder(batch_x_u_perturbed)

            weak_latent_z = Ts_self_model.projection_encoder(batch_x_u_weak)

            # calculate loss(L_r+ L_con)
            loss_con = info_nce_loss(strong_latent_z, weak_latent_z)
            loss_mask = bce_loss(mask_pred, batch_m_label)
            loss_feature = mse_loss(feature_pred, batch_x_u_weak)
            loss = loss_mask + alpha * loss_feature + beta * loss_con
            # loss = loss_mask + alpha * loss_feature
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x_u_weak.size(0)
        avg_loss = total_loss / N
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    # only return the encoder part
    return Ts_self_model.encoder
