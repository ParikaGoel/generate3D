import torch
import numpy as np


def bce(output, target):
    """
    
    :param output: output from the model of shape (N, D, H, W)
    :param target: ground truth of shape (N, D, H, W)
    :return: 
        mean bce loss for entire batch
    """
    batch_size = target.shape[0]
    output = torch.nn.Sigmoid()(output)
    criterion = torch.nn.BCELoss().cuda()
    loss = 0.0

    for i in range(batch_size):
        loss += criterion(output[i].view(-1), target[i].view(-1))
    
    loss = loss / batch_size

    return loss


def mse(output, target, batch_mask=None):
    batch_size = target.shape[0]
    assert (len(output.shape) > 1)
    criterion = torch.nn.MSELoss(reduction="none").cuda()
    loss = criterion(output, target)
    loss = torch.stack([torch.mean(loss[i]) for i in range(batch_size)])

    if batch_mask is not None:
        assert (len(batch_mask.shape) == 1)
        if sum(batch_mask) == 0:
            return torch.FloatTensor([0]).cuda()
        loss = torch.stack([loss[i] for i in range(batch_size) if batch_mask[i]])

    loss = torch.mean(loss)

    return loss

