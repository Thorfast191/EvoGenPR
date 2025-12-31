import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionAutoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU()
        )

        self.fc = nn.Linear(256 * 32 * 32, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 256 * 32 * 32)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        z_flat = z.view(z.size(0), -1)
        latent = self.fc(z_flat)

        recon = self.decoder_fc(latent)
        recon = recon.view(-1, 256, 32, 32)
        recon = self.decoder(recon)

        return recon, latent

