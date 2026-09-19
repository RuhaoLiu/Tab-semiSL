import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from con_mask import TsSelfModel
from vae_train import VAE
from augment_methods import vae_generate_data, gaussian_sample
from threshold_selector import DynamicThresholdEMA


class TsSemiSupervised(nn.Module):
    def __init__(self, input_dim, label_dim, hidden_dim):
        super(TsSemiSupervised, self).__init__()
        self.Linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.Linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.logits = nn.Linear(hidden_dim, label_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu1(self.Linear1(x))
        x = self.relu2(self.Linear2(x))
        y_hat_logit = self.logits(x)
        y_hat = self.softmax(y_hat_logit/3)
        return y_hat_logit


def con_mask_semi(x_train, y_train, x_unlab, x_test, parameters, gamma, p_m,
                  vae_file, encoder_file, device, datasets_name):
    """
    Semi-supervised components with self-adaptive dynamic threshold update mechanism.

    Args:
        - x_train, y_train: training data
        - x_unlab: unlabeled data
        - x_test: testing data
        - parameters: model parameters
        - gamma: control the weights of unsupervised learning
        - vae_file: vae file name
        - encoder_file: self encoder file name

    Returns:
        - y_test_hat: prediction on x_test
    """

    # Convert to DataFrame for consistent splitting
    x_train = pd.DataFrame(x_train.detach().cpu().numpy())

    # Network parameters
    vae_hidden_dim = parameters['vae_hidden_dim']
    vae_latent_dim = parameters['vae_latent_dim']
    predictor_hidden_dim = parameters['predictor_hidden_dim']
    batch_size = parameters['batch_size']
    iterations = parameters['iterations']
    label_dim = y_train.shape[1]

    # Train/validation split
    idx = np.random.permutation(x_train.shape[0])
    train_idx = idx[:int(0.9 * len(idx))]
    val_idx = idx[int(0.9 * len(idx)):]

    x_val, y_val = torch.tensor(x_train.iloc[val_idx, :].values, dtype=torch.float32, device=device), y_train[val_idx, :]
    x_train, y_train = torch.tensor(x_train.iloc[train_idx, :].values, dtype=torch.float32, device=device), y_train[train_idx, :]

    # Load VAE model
    VAE_model = VAE(input_dim=x_unlab.shape[1], hidden_dim=vae_hidden_dim, latent_dim=vae_latent_dim).to(device)
    VAE_model.load_state_dict(torch.load(vae_file))
    VAE_model.eval()

    # Data perturbation
    _, x_u_perturbed = vae_generate_data(VAE_model, x_unlab, p_m)
    # _, x_u_perturbed = gaussian_sample(x_unlab, p_m)
    x_u_weak = x_unlab
    x_u_perturbed = torch.tensor(x_u_perturbed, dtype=torch.float32, device=device)
    x_u_weak = torch.tensor(x_u_weak, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val, dtype=torch.float32, device=device)

    # Load encoder model
    dim = x_train.shape[1]
    proj_dim = 64
    encoder = TsSelfModel(dim, proj_dim).encoder.to(device)
    encoder.load_state_dict(torch.load(encoder_file))
    encoder.eval()

    with torch.no_grad():
        z = encoder(x_train)
        z_u_weak = encoder(x_u_weak)
        z_u_perturbed = encoder(x_u_perturbed)
        z_test = encoder(x_test)

    # Init predictor
    predictor = TsSemiSupervised(
        input_dim=x_train.shape[1],
        label_dim=label_dim,
        hidden_dim=predictor_hidden_dim
    ).to(device)

    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=False
    )

    # Training setup
    ce_loss = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(predictor.parameters(), lr=1e-3,
                           # weight_decay=1e-4
                           )

    best_val_loss = float('inf')
    # patience = int(iterations * 0.1)  # Early stopping patience
    patience = 50
    tiny = 0.1
    wait = 0
    warmup_step = 50
    softmax = nn.Softmax(dim=1)
    adaptive_threshold = DynamicThresholdEMA(label_dim, iterations)

    # save predictor
    predictor_dir = f'./save_model/predictor/{datasets_name}'
    os.makedirs(predictor_dir, exist_ok=True)
    predictor_file = os.path.join(predictor_dir, 'best_predictor.pth')

    # Training
    for step in range(iterations):
        predictor.train()

        # labeled data
        batch_idx = np.random.permutation(len(z))[:batch_size]
        z_batch, y_batch = z[batch_idx].to(device), y_train[batch_idx].to(device)

        # unlabeled data
        batch_u_idx = np.random.permutation(len(z_u_weak))[:batch_size]
        z_u_weak_batch = z_u_weak[batch_u_idx].to(device)
        z_u_strong_batch = z_u_perturbed[batch_u_idx].to(device)

        # all data
        inputs = torch.cat((z_batch, z_u_weak_batch, z_u_strong_batch))

        # Warm-up
        if step < warmup_step:
            # supervised loss
            logits = predictor(z_batch)
            loss_l = ce_loss(logits, torch.argmax(y_batch, dim=1))
            optimizer.zero_grad()
            loss_l.backward()
            optimizer.step()
            continue

        optimizer.zero_grad()

        # Supervised loss
        logits = predictor(inputs)
        logits_x_lb = logits[:z_batch.shape[0]]
        logits_u_weak, logits_u_strong = logits[z_batch.shape[0]:].chunk(2)
        loss_l = ce_loss(logits_x_lb, torch.argmax(y_batch, dim=1))

        # unsupervised loss with dynamic thresholding
        with torch.no_grad():
            max_probs, pseudo_labels = torch.max(softmax(logits_u_weak), dim=1)

            # Update threshold
            adaptive_threshold.update_thresholds(step, max_probs, pseudo_labels)
            current_thresholds = adaptive_threshold.thresholds.numpy()
            # print(current_thresholds)

            # select high-confidence samples
            sample_thresholds = current_thresholds[pseudo_labels.cpu().numpy()]
            selected_token = max_probs.cpu().numpy() > sample_thresholds
            selected_token = torch.tensor(selected_token, dtype=torch.float32, device=device)

            # Statistics
            # num_selected = selected_token.sum().item()
            # percentage = (num_selected / z_u_weak_batch.size(0)) * 100
            # if step % 100 == 0:
            #     print(f"Step {step}: Selected percentage: {percentage:.2f}%")

        # only selected samples calculate loss
        loss_u = ce_loss(logits_u_strong, pseudo_labels)
        loss_u = (loss_u * selected_token.float()).mean()

        # Total loss
        loss = loss_l + gamma * loss_u
        loss.backward()
        optimizer.step()

        # Validation
        predictor.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_val_batch, y_val_batch in val_loader:
                x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)
                z_val = encoder(x_val_batch)
                logits_val = predictor(z_val)
                val_loss += ce_loss(logits_val, torch.argmax(y_val_batch, dim=1)).item()

        avg_val_loss = val_loss / len(val_loader)
        # if step % 100 == 0 or step == iterations - 1:
            # print(f"Step {step}/{iterations}")
            # print(f"  Train Loss: {loss.item():.4f}")
            # print(f"  Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss + tiny:
            best_val_loss = avg_val_loss
            wait = 0
            torch.save(predictor.state_dict(), predictor_file)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break

    # Load best model and test
    predictor.load_state_dict(torch.load(predictor_file))
    predictor.eval()
    with torch.no_grad():
        y_test_hat = softmax(predictor(z_test))

    return y_test_hat.detach().cpu().numpy()