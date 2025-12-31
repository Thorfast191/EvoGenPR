class FeatureGrafting(nn.Module):
    def __init__(self, dim1, dim2, out_dim):
        super().__init__()
        self.fc = nn.Linear(dim1 + dim2, out_dim)

    def forward(self, f1, f2):
        fused = torch.cat([f1, f2], dim=1)
        return self.fc(fused)
