import torch
import torch.nn as nn
class SPFM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.fc(x)
    
    
class FeatureGrafting(nn.Module):
    def __init__(self, dim1, dim2, out_dim):
        super().__init__()
        self.fc = nn.Linear(dim1 + dim2, out_dim)

    def forward(self, f1, f2):
        fused = torch.cat([f1, f2], dim=1)
        return self.fc(fused)
