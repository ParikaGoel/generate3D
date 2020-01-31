import sys
sys.path.append("../.")
import torch
import config
import losses
import numpy as np
from PIL import Image
import voxel_grid as voxel
import dataset_loader as loader


def project(occ_grid, cam):
    transform = cam.extrinsic
    w = int(cam.resolution[0])
    h = int(cam.resolution[1])
    proj_img = np.full((h, w), 255, dtype=np.uint8)

    # occ_grid -> D x H x W
    positions = np.where(occ_grid == 1)
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
        u_max = min(511, int((cam.focal[0] * vertex_min[0]) / vertex_min[2] + cam.center[0]))
        v_max = min(511, int((cam.focal[1] * vertex_min[1]) / vertex_min[2] + cam.center[1]))

        u_min = max(0, int((cam.focal[0] * vertex_max[0]) / vertex_max[2] + cam.center[0]))
        v_min = max(0, int((cam.focal[1] * vertex_max[1]) / vertex_max[2] + cam.center[1]))

        proj_img[v_min:v_max, u_min:u_max] = 0

    proj_img[proj_img == 255] = 1
    proj_img = torch.from_numpy(proj_img).float()
    return proj_img

if __name__ == '__main__':
    gt_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-voxelized-gt/02747177/fd013bea1e1ffb27c31c70b1ddc95e3f__0__.txt"
    gt_img_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-renderings/02747177/fd013bea1e1ffb27c31c70b1ddc95e3f/color/color0.png"
    cam_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-renderings/02747177/fd013bea1e1ffb27c31c70b1ddc95e3f/cam/cam0.json"

    out_txt_file = "/home/parika/WorkingDir/complete3D/Assets/test.txt"
    out_ply_file = "/home/parika/WorkingDir/complete3D/Assets/test.ply"
    out = "/home/parika/WorkingDir/complete3D/Assets/"
    gt_occ = loader.load_sample(gt_file)
    gt_occ = gt_occ[0].numpy().transpose(2, 1, 0)

    obj_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-data/02747177/fd013bea1e1ffb27c31c70b1ddc95e3f/models/model_normalized.obj"
    # renderer.generate_images(obj_file, out, 10)

    gt_img = loader.load_img(gt_img_file)
    cam = loader.load_camera(cam_file)

    proj_img = project(gt_occ, cam)

    gt_img = gt_img.unsqueeze(0)
    proj_img = proj_img.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proj_loss = losses.vol_proj_loss(proj_img.to(device),
                                     gt_img.to(device), 1, device)
    print(proj_loss)

    proj_img = proj_img[0].numpy().astype(np.uint8)
    gt_img = gt_img[0].numpy().astype(np.uint8)
    proj_img[proj_img == 1] = 255
    gt_img[gt_img == 1] = 255

    img = Image.fromarray(proj_img, 'L')
    img.show()

    img = Image.fromarray(gt_img, "L")
    img.show()
