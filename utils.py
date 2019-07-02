import torch
from sklearn.metrics import cohen_kappa_score


def quadratic_kappa(y_hat, y):
    return torch.tensor(cohen_kappa_score(y_hat, y, weights='quadratic'))
