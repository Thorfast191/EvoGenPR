import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    accuracy_score
)

def multilabel_accuracy(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred >= threshold).astype(int)
    return accuracy_score(y_true, y_pred_bin)

def multilabel_precision(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred >= threshold).astype(int)
    return precision_score(
        y_true, y_pred_bin,
        average="macro",
        zero_division=0
    )

def multilabel_auc(y_true, y_pred):
    aucs = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) > 0:
            aucs.append(
                roc_auc_score(y_true[:, i], y_pred[:, i])
            )
    return float(np.mean(aucs))
