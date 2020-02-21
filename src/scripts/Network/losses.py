import torch
import numpy as np


def create_target_mask2(output, target, weight):
    weights = target.data.clone()

    output_inds = output <= 0.1
    target_inds = target == 0
    intersection = output_inds & target_inds

    weights.fill_(weight)
    weights.masked_fill_(intersection, 1)
    return weights


def create_target_mask(target, weight):
    weights = target.data.clone()
    weights[weights > 0] = weight
    weights[weights == 0] = 1
    return weights
    
    
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


def proj_loss2(output, target, device):
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
    # loss = bce_loss(output, target)
    criterion = torch.nn.MSELoss(reduction="none").to(device)
    loss = criterion(output.float(), target.float())
    # loss = torch.stack([torch.mean(torch.stack([torch.sum(loss[b,v]) for v in range(n_views)]))
    #                     for b in range(batch_size)])
    #
    # loss = torch.mean(loss)

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
    # loss = bce_loss(output, target)
    criterion = torch.nn.MSELoss(reduction="none").to(device)
    loss = criterion(output.float(), target.float())
    loss = torch.stack([torch.mean(torch.stack([torch.sum(loss[b,v]) for v in range(n_views)]))
                        for b in range(batch_size)])

    loss = torch.mean(loss)

    return loss