import torch


def mean_absolute_error(preds, targets):
    return torch.mean(torch.abs(preds - targets))


def mean_squared_error(preds, targets):
    return torch.mean((preds - targets) ** 2)


def accuracy(preds, targets):
    _, preds = torch.max(preds, 1)
    return (preds == targets).float().mean()
