import sys
sys.path.append("../.")
import config
import torch
import losses
import renderer
import pyrender
import voxel_grid as voxel
from PIL import Image
import eval_metric
import numpy as np
import dataset_loader as loader
import torch.nn.functional as F

def project(gt_img_file, occ_grid, transform):
    w, h = 512, 512
    center = w/2
    data = np.full((h, w), 255, dtype=np.uint8)
    mapping = np.full((h,w,3), 33, dtype=np.uint8) # index mapping between voxel and image
    color = [255, 0, 0]

    # occ_grid -> D x H x W
    positions = np.where(occ_grid == 1)
    min_bound = np.array([-0.5, -0.5, -0.5])
    voxel_scale = 1/ 32
    for i, j, k in zip(*positions):
        vertex_min = (np.array([-1.0, -1.0, -1.0]) + np.array([i, j, k])).astype(float)
        vertex_max = (np.array([1.0, 1.0, 1.0]) + np.array([i, j, k])).astype(float)
        # vertex = np.array([i, j, k]).astype(float)
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
        u_max = min(511,int((config.focal * vertex_min[0]) / vertex_min[2] + center))
        v_max = min(511,int((config.focal * vertex_min[1]) / vertex_min[2] + center))

        u_min = max(0,int((config.focal * vertex_max[0]) / vertex_max[2] + center))
        v_min = max(0,int((config.focal * vertex_max[1]) / vertex_max[2] + center))

        data[v_min:v_max,u_min:u_max] = 0
        mapping[v_min:v_max,u_min:u_max,] = [i, j, k]

    img = Image.fromarray(data, 'L')
    img.show()

    gt_img = Image.open(gt_img_file)
    img = gt_img.convert("L")
    img_np = np.array(img)
    img_np[img_np < 255] = 0
    img = Image.fromarray(img_np, "L")
    # gt = np.full((h, w), 255, dtype=np.uint8)
    img.show()

if __name__ == '__main__':
    gt_file = "/home/parika/WorkingDir/complete3D/Assets_remote/shapenet-voxelized-gt/02747177/fd013bea1e1ffb27c31c70b1ddc95e3f__0__.txt"
    gt_img_file = "/home/parika/WorkingDir/complete3D/Assets_remote/shapenet-renderings/02747177/fd013bea1e1ffb27c31c70b1ddc95e3f/color/color0.png"
    out_txt_file = "/home/parika/WorkingDir/complete3D/Assets/test.txt"
    out_ply_file = "/home/parika/WorkingDir/complete3D/Assets/test.ply"
    out = "/home/parika/WorkingDir/complete3D/Assets/"
    gt_occ = loader.load_sample(gt_file)
    gt_occ = gt_occ[0].numpy().transpose(2, 1, 0)

    obj_file = "/home/parika/WorkingDir/complete3D/Assets_remote/shapenet-data/02747177/fd013bea1e1ffb27c31c70b1ddc95e3f/models/model_normalized.obj"
    # renderer.generate_images(obj_file, out, 10)



    transform = np.eye(4)
    transform[2, 3] = 1.2
    transform = np.linalg.inv(transform)
    print(transform)

    project(gt_img_file, gt_occ, transform)
    # transform = np.linalg.inv(transform)
    # transform = transform[:3, ]
    # transform = torch.from_numpy(transform)
    # transform = transform.reshape(1, 3, 4)

    # grid = F.affine_grid(transform, gt_occ.size())
    # out_occ = F.grid_sample(gt_occ, grid)
    # out_occ = out_occ[0]
    #
    # loader.save_sample(out_txt_file, out_occ)
    # voxel_grid.txt_to_mesh(out_txt_file, out_ply_file)
