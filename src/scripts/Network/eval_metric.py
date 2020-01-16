import torch
import dataset_loader as dataloader


def iou(pred, target):
    """
    calculates the IOU metric (Intersection over Union) for a test prediction by the network
    :param pred: predicted output by the network
    :param target: corresponding ground truth
    :return: returns the IOU value
    """
    pred = pred.view(-1)
    target = target.view(-1)

    pred_inds = pred == 1
    target_inds = target == 1
    intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
    iou = float(intersection) / float(max(union, 1))

    return iou


if __name__ == '__main__':
    output_file = "/home/parika/WorkingDir/complete3D/Assets/output-network/output.txt"
    target_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-voxelized-gt/04379243/142060f848466cad97ef9a13efb5e3f7__0__.txt"

    pred = dataloader.load_sample(output_file)
    target = dataloader.load_sample(target_file)

    error = iou(pred, target)
    print(error)
