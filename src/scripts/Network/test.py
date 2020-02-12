import torch
import numpy as np
from PIL import Image
import dataset_loader as loader
import perspective_projection as projection

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder = "/media/sda2/shapenet/test/fd013bea1e1ffb27c31c70b1ddc95e3f/pose/"
    gt_file = "/media/sda2/shapenet/test/fd013bea1e1ffb27c31c70b1ddc95e3f__test__.txt"
    gt_occ = loader.load_sample(gt_file).to(device)
    poses = loader.load_poses(folder).to(device)

    projection_helper = projection.ProjectionHelper()
    lin_index_map = torch.stack([projection_helper.compute_projection(gt_occ, pose) for pose in poses])
    proj_imgs = projection_helper.forward(gt_occ.unsqueeze(0), lin_index_map.unsqueeze(0))
    proj_imgs = proj_imgs[0]

    for proj_img in proj_imgs:
        projection_helper.show_projection(proj_img)

    # occ_grid = projection_helper.backward(proj_imgs, lin_index_map, gt_occ)

