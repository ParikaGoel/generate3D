import torch
import numpy as np
from PIL import Image
import dataset_loader as loader
import perspective_projection as projection

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = "../../../Assets_remote/test/fd013bea1e1ffb27c31c70b1ddc95e3f/pose/"
    gt_file = "../../../Assets_remote/test/fd013bea1e1ffb27c31c70b1ddc95e3f__test__.txt"
    gt_occ = loader.load_sample(gt_file).to(device)
    poses = loader.load_poses(folder).to(device)
    poses[:,11] = -1.2
    poses = torch.stack([torch.reshape(pose, (4,4)) for pose in poses])

    projection_helper = projection.ProjectionHelper()
    lin_index_map = torch.stack([projection_helper.compute_projection(gt_occ, pose) for pose in poses])
    proj_imgs = projection_helper.forward(gt_occ, lin_index_map)

    occ_grid = projection_helper.backward(proj_imgs, lin_index_map, gt_occ)



