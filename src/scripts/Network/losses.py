import torch
import numpy as np


def create_target_mask(target, weight):
    weights = target.data.clone()
    weights[weights > 0] = weight
    weights[weights == 0] = 1
    return weights
    

def weighted_l1(device, output, target, trunc_dist, weight):
    batch_size = target.shape[0]
    assert (len(output.shape) > 1)

    weights = torch.empty(target.shape).fill_(0)
    mask = torch.gt(target, 0) & torch.le(target, trunc_dist)
    weights[mask] = weight

    criterion = torch.nn.L1Loss(reduction="none").to(device)
    loss = criterion(output.float(), target.float())
    loss = loss * weights
    loss = torch.stack([torch.mean(loss[i]) for i in range(batch_size)])

    loss = torch.mean(loss)

    return loss


def l1(output, target, device):
    batch_size = target.shape[0]
    assert (len(output.shape) > 1)
    criterion = torch.nn.L1Loss(reduction="none").to(device)
    loss = criterion(output.float(), target.float())
    loss = torch.stack([torch.mean(loss[i]) for i in range(batch_size)])

    loss = torch.mean(loss)

    return loss


def mse(output, target, device):
    batch_size = target.shape[0]
    assert (len(output.shape) > 1)
    criterion = torch.nn.MSELoss(reduction="none").to(device)
    loss = criterion(output.float(), target.float())
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
    loss = criterion(output.float(), target.float())
    loss = torch.stack([torch.mean(loss[i]) for i in range(batch_size)])

    loss = torch.mean(loss)

    return loss


def weighted_bce(output, target, weight, device):
    batch_size = target.shape[0]
    assert(len(output.shape) > 1)

    output = torch.nn.Sigmoid()(output)
    weights = create_target_mask(target, weight)
    criterion = torch.nn.BCELoss(weight=weights, reduction="none").to(device)
    loss = criterion(output, target)
    loss = torch.stack([torch.mean(loss[i]) for i in range(batch_size)]).to(device)

    loss = torch.mean(loss)

    return loss


def proj_loss(output, target, device):
    """
    Computes the projection loss
    :param output: Predicted voxel projected into image space ; shape : (N, n_views, (img_w * img_h))
    :param target: Target Image Silhouette
    :param weight: Weight for projection loss
    :param device: cpu or cuda
    :return:
        MSE loss between the ground truth masks (object silhouettes)
        and the predicted masks
    """
    batch_size = target.shape[0]
    n_views = target.shape[1]
    # output = torch.nn.Sigmoid()(output)
    criterion = torch.nn.BCELoss(reduction="none").to(device)
    loss = criterion(output.float(), target.float())
    loss = torch.stack([torch.mean(torch.stack([torch.mean(loss[b,v]) for v in range(n_views)]))
                        for b in range(batch_size)])

    loss = torch.mean(loss)

    return loss