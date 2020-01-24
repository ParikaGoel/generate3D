import torch
import numpy as np


def bce(output, target, device):
    """
    
    :param output: output from the model of shape (N, D, H, W)
    :param target: ground truth of shape (N, D, H, W)
    :return: 
        mean bce loss for entire batch
    """
    batch_size = target.shape[0]
    output = torch.nn.Sigmoid()(output)
    criterion = torch.nn.BCELoss().to(device)
    loss = criterion(output, target)

    # for i in range(batch_size):
    #     loss += criterion(output[i].view(-1), target[i].view(-1))
    #
    # loss = loss / batch_size
    loss = torch.mean(loss)

    return loss


def mse(output, target, device):
    batch_size = target.shape[0]
    assert (len(output.shape) > 1)
    criterion = torch.nn.MSELoss(reduction="none").to(device)
    loss = criterion(output, target)
    loss = torch.stack([torch.mean(loss[i]) for i in range(batch_size)])

    loss = torch.mean(loss)

    return loss

