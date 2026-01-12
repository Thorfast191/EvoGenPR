import torch.nn as nn
import torchvision.models as models


def resnet_encoder():
    """
    Returns a ResNet-18 encoder with output dim = 512
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    return model


def swin_encoder():
    """
    Returns a Swin-T encoder with output dim = 768
    """
    model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
    model.head = nn.Identity()
    return model