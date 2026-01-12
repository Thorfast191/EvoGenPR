import torch.nn as nn
class SENPR(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = resnet_encoder()
        self.swin = swin_encoder()

        self.fusion = FeatureGrafting(512, 768, 512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        f1 = self.resnet(x)
        f2 = self.swin(x)
        fused = self.fusion(f1, f2)
        return self.classifier(fused)
