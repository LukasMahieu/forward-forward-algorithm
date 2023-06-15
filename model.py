import torch


def goodness(y, theta):
    return torch.sigmoid(torch.sum(torch.square(y) - theta))
