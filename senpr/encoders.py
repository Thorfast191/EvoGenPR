import timm
import torchvision.models as models

def resnet_encoder():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()
    return model

def swin_encoder():
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=True,
        num_classes=0
    )
    return model
