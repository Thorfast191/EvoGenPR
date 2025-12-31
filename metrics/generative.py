import torch
import numpy as np
from torchvision.models import inception_v3
from scipy.linalg import sqrtm

@torch.no_grad()
def extract_features(images, device):
    model = inception_v3(
        pretrained=True,
        transform_input=False
    ).to(device)
    model.eval()
    model.fc = torch.nn.Identity()

    feats = []
    for img in images:
        img = img.to(device)
        f = model(img.unsqueeze(0))
        feats.append(f.cpu().numpy())

    return np.concatenate(feats, axis=0)

def calculate_fid(real_images, fake_images, device):
    real_feats = extract_features(real_images, device)
    fake_feats = extract_features(fake_images, device)

    mu1, sigma1 = real_feats.mean(0), np.cov(real_feats, rowvar=False)
    mu2, sigma2 = fake_feats.mean(0), np.cov(fake_feats, rowvar=False)

    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum((mu1 - mu2) ** 2) + np.trace(
        sigma1 + sigma2 - 2 * covmean
    )
    return float(fid)

def inception_score(images, device, splits=10):
    model = inception_v3(pretrained=True).to(device)
    model.eval()

    preds = []
    for img in images:
        p = torch.softmax(
            model(img.unsqueeze(0).to(device)), dim=1
        )
        preds.append(p.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    scores = []

    for i in range(splits):
        part = preds[i::splits]
        kl = part * (np.log(part) - np.log(part.mean(0)))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))

    return float(np.mean(scores))
