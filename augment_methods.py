import torch
from utils import mask_generator, pretext_generator
import numpy as np


def vae_generate_data(model, data_tensor, p_m):

    augmented_samples = []
    with torch.no_grad():
        for x in data_tensor:
            x = x.view(1, -1)  # [1, D]
            mu, logvar = model.encode(x)
            z = model.reparameterize(mu, logvar)
            x_hat = model.decode(z)
            augmented_samples.append(x_hat.squeeze(0))

    x_decode = torch.stack(augmented_samples, dim=0)  # [N, D]
    m = mask_generator(p_m, x_decode.cpu().numpy())
    m_label, x_prime = pretext_generator(m, x_decode.cpu().numpy())
    return m_label, x_prime


def gaussian_sample(data, p_m, random_state=42):

    np.random.seed(random_state)
    if isinstance(data, np.ndarray):
        noise = np.random.normal(0, 0.1, data.shape)
        data_noisy = data + noise
    elif isinstance(data, torch.Tensor):
        noise = torch.randn_like(data) * 0.1 + 0
        data_noisy = data + noise
    # sigma = 0.1
    # noise = np.random.normal(loc=0.0, scale=sigma, size=data.shape)
    # data_noisy = data.cpu() + noise.cpu()
    m = mask_generator(p_m, data_noisy)
    m_label, x_prime = pretext_generator(m, data_noisy.cpu().numpy())
    return m_label, x_prime