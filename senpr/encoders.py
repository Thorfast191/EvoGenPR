
import torch.nn as nn
import torchvision.models as models


def resnet_encoder():
    """
    ResNet-18 encoder
    Output: [B, 512]
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    return model


def swin_encoder():
    """
    Swin-T encoder
    Output: [B, 768]
    """
    model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    model.head = nn.Identity()
    return model