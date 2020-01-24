import torch
import losses
import eval_metric
import numpy as np
import dataset_loader as loader

if __name__ == '__main__':
    gt_file = "/home/parika/WorkingDir/complete3D/Assets_remote/shapenet-voxelized-gt/02747177/fd013bea1e1ffb27c31c70b1ddc95e3f__0__.txt"
    gt_occ = loader.load_sample(gt_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iou = eval_metric.iou(gt_occ, gt_occ)
    loss = losses.bce(gt_occ, gt_occ, device)
    print(iou, "  ", loss)
