import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weighted_l1(output, target, trunc_dist, weight):
    batch_size = target.shape[0]
    assert (len(output.shape) > 1)

    weights = torch.empty(target.shape).fill_(1).to(device)
    mask = torch.ge(target, 0) & torch.lt(target, trunc_dist)
    weights[mask] = weight

    criterion = torch.nn.L1Loss(reduction="none").to(device)
    loss = criterion(output.float(), target.float())
    loss = loss * weights
    loss = torch.stack([torch.mean(loss[i]) for i in range(batch_size)])

    loss = torch.mean(loss)

    return loss


def apply_log_transform(df):
    sgn = torch.sign(df)
    out = torch.log(torch.abs(df) + 1)
    out = sgn * out
    return out


def l1(output_df, target_df, use_log_transform=False):
    batch_size = target_df.shape[0]
    assert (len(output_df.shape) > 1)
    criterion = torch.nn.L1Loss(reduction="none").to(device)

    if use_log_transform:
        output_df = apply_log_transform(output_df)
        target_df = apply_log_transform(target_df)
    
    loss = criterion(output_df.float(), target_df.float())
    loss = torch.stack([torch.mean(loss[i]) for i in range(batch_size)])

    loss = torch.mean(loss)

    return loss


def bce(output_occ, target_occ):
    """
    
    :param output: output from the model of shape (N, D, H, W)
    :param target: ground truth of shape (N, D, H, W)
    :return: 
        mean bce loss for entire batch
    """
    batch_size = target_occ.shape[0]
    output_occ = torch.nn.Sigmoid()(output_occ)
    criterion = torch.nn.BCELoss(reduction="none").to(device)
    loss = criterion(output_occ.float(), target_occ.float())
    loss = torch.stack([torch.mean(loss[i]) for i in range(batch_size)])

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
    criterion = torch.nn.MSELoss(reduction="none").to(device)
    loss = criterion(output.float(), target.float())
    loss = torch.stack([torch.mean(torch.stack([torch.mean(loss[b, v]) for v in range(n_views)]))
                        for b in range(batch_size)])

    loss = torch.mean(loss)

    return loss
