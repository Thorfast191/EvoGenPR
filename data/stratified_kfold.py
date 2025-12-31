import numpy as np
from collections import defaultdict

def make_folds(X, Y, n_splits=5, seed=42):
    """
    Custom multilabel stratified K-Fold.
    Approximates label distribution balance across folds.
    
    Parameters:
    - X: array-like indices
    - Y: binary label matrix [n_samples, n_classes]
    """

    rng = np.random.RandomState(seed)
    n_samples, n_classes = Y.shape

    # Count label frequency
    label_freq = Y.sum(axis=0)

    # Sort samples by label rarity (important!)
    sample_order = np.argsort(
        -np.sum(Y / (label_freq + 1e-6), axis=1)
    )

    folds = [[] for _ in range(n_splits)]
    fold_label_counts = np.zeros((n_splits, n_classes))

    for idx in sample_order:
        # Assign sample to fold with lowest label imbalance
        fold_scores = []

        for f in range(n_splits):
            new_counts = fold_label_counts[f] + Y[idx]
            score = np.sum(new_counts / (label_freq + 1e-6))
            fold_scores.append(score)

        best_fold = np.argmin(fold_scores)
        folds[best_fold].append(idx)
        fold_label_counts[best_fold] += Y[idx]

    # Convert to (train_idx, val_idx)
    all_indices = set(range(n_samples))
    splits = []

    for f in range(n_splits):
        val_idx = np.array(folds[f])
        train_idx = np.array(list(all_indices - set(val_idx)))
        splits.append((train_idx, val_idx))

    return splits
