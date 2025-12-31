class GANGenerator(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256*16*16),
            nn.ReLU(),
            nn.Unflatten(1, (256,16,16)),
            nn.ConvTranspose2d(256,128,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,1,4,2,1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.net(z)
