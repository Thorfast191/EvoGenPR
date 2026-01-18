import torch.nn as nn

from senpr.encoders import resnet_encoder, swin_encoder
from senpr.fusion import FeatureGrafting, SPFM


class SENPR(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Encoders
        self.resnet = resnet_encoder()   # outputs [B, 512]
        self.swin = swin_encoder()       # outputs [B, 768]

        # 🔹 SPFM: per-stream feature refinement
        self.spfm_resnet = SPFM(512)
        self.spfm_swin = SPFM(768)

        # 🔹 Feature fusion
        self.fusion = FeatureGrafting(512, 768, 512)

        # Classifier
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # Encode
        f1 = self.resnet(x)   # [B, 512]
        f2 = self.swin(x)     # [B, 768]

        # 🔹 Refine features with SPFM
        f1 = self.spfm_resnet(f1)
        f2 = self.spfm_swin(f2)

        # Fuse
        fused = self.fusion(f1, f2)  # [B, 512]

        # Classify
        return self.classifier(fused)