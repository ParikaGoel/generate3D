import sys
sys.path.append("../.")
import torch
import config
import losses
import numpy as np
from PIL import Image
import voxel_grid as voxel
import dataset_loader as loader


def project_batch(occ_grids, transforms):
    batch_size = occ_grids.shape[0]
    projs = []
    for i in range(batch_size):
        proj_img = project(occ_grids[i], transforms[i])
        projs.append(proj_img)

    proj_imgs = torch.stack(projs, dim=0)
    return proj_imgs


def project(occ_grid, transform):
    w = config.render_img_width
    h = config.render_img_height
    center = w / 2

    occ_grid = occ_grid[0].transpose(2, 0)  # removes the channel dimension and changes shape to [W, H, D]
    proj_img = torch.empty((h, w), dtype=torch.float).fill_(1.0)

    # occ_grid -> D x H x W
    positions = torch.where(occ_grid == 1)
    min_bound = np.array([-0.5, -0.5, -0.5])
    voxel_scale = 1/ 32
    for i, j, k in zip(*positions):
        vertex_min = (np.array([-1.0, -1.0, -1.0]) + np.array([i, j, k])).astype(float)
        vertex_max = (np.array([1.0, 1.0, 1.0]) + np.array([i, j, k])).astype(float)
        vertex_min *= voxel_scale
        vertex_min += min_bound

        vertex_max *= voxel_scale
        vertex_max += min_bound

        if transform is not None:
            rotation = transform[0:3, 0:3]
            translation = transform[0:3, 3]
            vertex_min = np.matmul(rotation, vertex_min) + translation
            vertex_max = np.matmul(rotation, vertex_max) + translation

        #project onto image space
        u_max = min(511, int((config.focal * vertex_min[0]) / vertex_min[2] + center))
        v_max = min(511, int((config.focal * vertex_min[1]) / vertex_min[2] + center))

        u_min = max(0, int((config.focal * vertex_max[0]) / vertex_max[2] + center))
        v_min = max(0, int((config.focal * vertex_max[1]) / vertex_max[2] + center))

        proj_img[v_min:v_max, u_min:u_max] = 0.0

    return proj_img


if __name__ == '__main__':
    gt_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-voxelized-gt/02747177/fd013bea1e1ffb27c31c70b1ddc95e3f__0__.txt"
    gt_img_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-renderings/02747177/fd013bea1e1ffb27c31c70b1ddc95e3f/color/color0.png"
    cam_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-renderings/02747177/fd013bea1e1ffb27c31c70b1ddc95e3f/cam/cam0.json"

    out_txt_file = "/home/parika/WorkingDir/complete3D/Assets/test.txt"
    out_ply_file = "/home/parika/WorkingDir/complete3D/Assets/test.ply"
    out = "/home/parika/WorkingDir/complete3D/Assets/"
    gt_occ = loader.load_sample(gt_file)

    obj_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-data/02747177/fd013bea1e1ffb27c31c70b1ddc95e3f/models/model_normalized.obj"
    # renderer.generate_images(obj_file, out, 10)

    gt_img = loader.load_img(gt_img_file)
    transform = loader.get_extrinsic(cam_file)

    gt_occ = gt_occ.unsqueeze(0)
    transform = np.expand_dims(transform, 0)
    # proj_img = project(gt_occ, transform)
    proj_img = project_batch(gt_occ, transform)

    gt_img = gt_img.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proj_loss = losses.vol_proj_loss(proj_img.float().to(device),
                                     gt_img.float().to(device), 1, device)
    print(proj_loss)

    proj_img = proj_img[0].numpy().astype(np.uint8)
    gt_img = gt_img[0].numpy().astype(np.uint8)
    proj_img[proj_img == 1] = 255
    gt_img[gt_img == 1] = 255

    img = Image.fromarray(proj_img, 'L')
    img.show()

    img = Image.fromarray(gt_img, "L")
    img.show()
