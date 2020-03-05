def iou(pred, target):
    """
    calculates the IOU metric (Intersection over Union) for a test prediction by the network
    :param pred: predicted output by the network
    :param target: corresponding ground truth
    :return: returns the IOU value
    """
    pred = pred.view(-1)
    target = target.view(-1)

    pred_inds = pred > 0.5
    target_inds = target == 1
    intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
    iou = float(intersection) / float(max(union, 1))

    return iou