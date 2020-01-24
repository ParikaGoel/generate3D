import torch
import numpy as np


def create_target_mask(target, weight):
    weights = target.data.clone()
    weights[weights > 0] = weight
    weights[weights == 0] = 1
    return weights
    
    
def mse(output, target, device):
    batch_size = target.shape[0]
    assert (len(output.shape) > 1)
    criterion = torch.nn.MSELoss(reduction="none").to(device)
    loss = criterion(output, target)
    loss = torch.stack([torch.mean(loss[i]) for i in range(batch_size)])

    loss = torch.mean(loss)

    return loss


def bce(output, target, device):
    """
    
    :param output: output from the model of shape (N, D, H, W)
    :param target: ground truth of shape (N, D, H, W)
    :return: 
        mean bce loss for entire batch
    """
    batch_size = target.shape[0]
    output = torch.nn.Sigmoid()(output)
    criterion = torch.nn.BCELoss(reduction="none").to(device)
    loss = criterion(output, target)
    loss = torch.stack([torch.mean(loss[i]) for i in range(batch_size)])

    loss = torch.mean(loss)

    return loss


def weighted_bce(output, target, weight, device):
    batch_size = target.shape[0]
    assert(len(output.shape) > 1)

    weights = create_target_mask(target, weight)
    output = torch.nn.Sigmoid()(output)
    criterion = torch.nn.BCELoss(weight=weights, reduction="none").to(device)
    loss = criterion(output, target)
    loss = torch.stack([torch.mean(loss[i]) for i in range(batch_size)]).to(device)

    loss = torch.mean(loss)

    return loss

