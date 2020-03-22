import torch


def iou_occ(pred, target):
    """
    calculates the IOU metric (Intersection over Union) for a test prediction by the network
    :param pred: predicted output by the network
    :param target: corresponding ground truth
    :return: returns the IOU value
    """
    pred = torch.nn.Sigmoid()(pred)
    pred = pred.view(-1)
    target = target.view(-1)

    pred_inds = pred > 0.5
    target_inds = target == 1
    intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
    iou = float(intersection) / float(max(union, 1))

    return iou


def iou_df(pred, target, trunc_dist):
    """
    calculates the IOU metric (Intersection over Union) for a test prediction by the network
    :param pred: predicted output by the network
    :param target: corresponding ground truth
    :return: returns the IOU value
    """
    pred = pred.view(-1)
    target = target.view(-1)

    pred_mask = torch.ge(pred, 0.0) & torch.lt(pred, trunc_dist)
    target_mask = torch.ge(target, 0.0) & torch.lt(target, trunc_dist)

    intersection = pred_mask & target_mask
    intersection_count = (intersection[intersection == True]).long().sum().data.cpu().item()
    union = (pred_mask[pred_mask == True]).long().sum().data.cpu().item() + (target_mask[target_mask == True]).long().sum().data.cpu().item() - intersection_count
    iou = float(intersection_count) / float(max(union, 1))

    return iou


def iou_sdf(pred, target, trunc_dist = 1.0):
    """
    calculates the IOU metric (Intersection over Union) for a test prediction by the network
    :param pred: predicted output by the network
    :param target: corresponding ground truth
    :return: returns the IOU value
    """
    pred = pred.view(-1)
    target = target.view(-1)

    pred_mask = torch.ge(pred, -trunc_dist) & torch.le(pred, trunc_dist)
    target_mask = torch.ge(target, -trunc_dist) & torch.le(target, trunc_dist)

    intersection = pred_mask & target_mask
    intersection_count = (intersection[intersection == True]).long().sum().data.cpu().item()
    union = (pred_mask[pred_mask == True]).long().sum().data.cpu().item() + (target_mask[target_mask == True]).long().sum().data.cpu().item() - intersection_count
    iou = float(intersection_count) / float(max(union, 1))

    return iou
