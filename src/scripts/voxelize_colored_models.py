import sys
sys.path.append('./Network')
import os
import torch
import config
import JSONHelper
import voxel_grid
import numpy as np
from camera import *
from PIL import Image
import dataset_loader as loader

params = JSONHelper.read("./parameters.json")

def make_intrinsic():
    intrinsic = torch.eye(4)
    intrinsic[0][0] = config.focal
    intrinsic[1][1] = config.focal
    intrinsic[0][2] = config.render_img_width / 2
    intrinsic[1][2] = config.render_img_height / 2
    return intrinsic

def make_world_to_grid():
    world_to_grid = torch.eye(4)
    world_to_grid[0][3] = world_to_grid[1][3] = world_to_grid[2][3] = 0.5
    world_to_grid *= config.vox_dim
    world_to_grid[3][3] = 1.0
    return world_to_grid


def compute_index_mapping(world_to_camera, grid_to_world, device):
    lin_ind_volume = torch.arange(0, config.vox_dim * config.vox_dim * config.vox_dim,
                                  out=torch.LongTensor()).to(device)
    grid_coords = grid_to_world.new(4, lin_ind_volume.size(0)).int()
    grid_coords[2] = lin_ind_volume / (config.vox_dim * config.vox_dim)
    tmp = lin_ind_volume - (grid_coords[2] * config.vox_dim * config.vox_dim).long()
    grid_coords[1] = tmp / config.vox_dim
    grid_coords[0] = torch.remainder(tmp, config.vox_dim)
    grid_coords[3].fill_(1)

    # cube vertex coords
    vertex_coords = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1],
                                  [0, 0, 1, 1, 0, 0, 1, 1],
                                  [0, 0, 0, 0, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0]]).to(device)

    corners = grid_coords.unsqueeze(0).repeat(8, 1, 1)
    for i in range(8):
        corners[i] = corners[i] + vertex_coords[:, i, None]

    # transform to current frame
    camera_coords = torch.matmul(world_to_camera, torch.matmul(grid_to_world, corners.float()))

    # project into image
    intrinsic = make_intrinsic().to(device)
    p = camera_coords.clone()
    for i in range(8):
        p[i, 1, :] = -p[i, 1, :]  # perspective projection flips the image in y- direction
        p[i, 0] = (p[i, 0] * intrinsic[0][0]) / torch.abs(p[i, 2]) + intrinsic[0][2]
        p[i, 1] = (p[i, 1] * intrinsic[1][1]) / torch.abs(p[i, 2]) + intrinsic[1][2]
    p = torch.round(p).long()

    p = torch.clamp(p, min=0, max=511)
    index_map = p.new(4, p.size(2))
    index_map[:2, :] = torch.min(p, dim=0).values[0:2, :]
    index_map[2:, :] = torch.max(p, dim=0).values[0:2, :]

    lin_index_map = torch.flatten(index_map, start_dim=0, end_dim=-1)

    return lin_index_map


def save_vox_as_txt(txt_file, voxel_grid, occ_grid=None):
    voxel_grid = voxel_grid.cpu().numpy().astype(np.uint16)
    voxel_grid = voxel_grid.reshape((config.vox_dim, config.vox_dim, config.vox_dim, 3)).transpose(2, 1, 0, 3)

    if occ_grid is not None:
        occ_grid = occ_grid.cpu().numpy().transpose(2, 1, 0)
        positions = np.argwhere(occ_grid == 1)
    else:
        positions = np.argwhere(np.all(voxel_grid < 256, axis=3))

    with open(txt_file, "w") as f:
        for i, j, k in positions:
            color = voxel_grid[i, j, k]
            data = np.column_stack((i, j, k, color[0], color[1], color[2]))
            np.savetxt(f, data, fmt='%d %d %d %d %d %d', delimiter=' ')


def raymarch(voxel_grid, occ_grid, color_file, depth_file, pose_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    occ_grid = torch.flatten(occ_grid)
    grid_to_world = torch.inverse(make_world_to_grid()).to(device)

    im = Image.open(color_file)
    color = torch.from_numpy(np.array(im)).float().to(device)

    depth_im = Image.open(depth_file)
    depth = torch.from_numpy(np.array(depth_im)).float().to(device)

    pose = loader.load_pose(pose_file).to(device)
    index_map = compute_index_mapping(pose, grid_to_world, device).to(device)

    index_map = index_map.reshape((4, -1))

    for i in range(voxel_grid.size(0)):
        if occ_grid[i] == 1:
            c = color[index_map[1, i]:index_map[3, i], index_map[0, i]:index_map[2, i]]
            m = torch.gt(depth[index_map[1, i]:index_map[3, i], index_map[0, i]:index_map[2, i]], 0)
            if m.any():
                valid_colors = c[m]
                mean_color = torch.mean(valid_colors, axis=0)
                voxel_grid[i] = mean_color

    voxel_grid = torch.ceil(voxel_grid)
    # voxel_grid = torch.ceil(voxel_grid).cpu().numpy().astype(np.uint16)

    return voxel_grid


def raymarch_n_views(color_img_dir, depth_img_dir, pose_dir, occ_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_files = sorted(os.listdir(color_img_dir))
    depth_files = sorted(os.listdir(depth_img_dir))
    pose_files = sorted(os.listdir(pose_dir))
    occ_grid = loader.load_sample(occ_file)[0]

    voxel_grid = torch.empty((config.vox_dim * config.vox_dim * config.vox_dim, 3),
                             dtype=torch.float).fill_(256).to(device)

    for i in range(len(img_files)):
        voxel_grid = raymarch(voxel_grid, occ_grid, color_img_dir + img_files[i], depth_img_dir + depth_files[i],
                              pose_dir + pose_files[i])

    # voxel_grid = raymarch(voxel_grid, occ_grid, color_img_dir + img_files[0], depth_img_dir + depth_files[0],
    #                       pose_dir + pose_files[0])
    # grid = torch.stack([raymarch(voxel_grid, occ_grid, color_img_dir+img_files[i], depth_img_dir+depth_files[i], pose_dir+pose_files[i]) for i in range(len(img_files))])
    #
    # voxel_grid = torch.empty((config.vox_dim * config.vox_dim * config.vox_dim, 3),
    #                          dtype=torch.float).fill_(256).to(device)
    #
    # for i in range(grid.size(1)):
    #     colors = grid[:,i,:]
    #     mask = torch.lt(colors, 256)
    #     if mask.any():
    #         voxel_grid[i] = torch.max(colors[mask].reshape(-1,3), axis=0).values

    # voxel_grid = torch.ceil(voxel_grid)

    return voxel_grid


def voxelize(synset_id, model_id):
    color_img_dir = params["shapenet_renderings"] + synset_id + "/" + model_id + "/color/"
    depth_img_dir = params["shapenet_renderings"] + synset_id + "/" + model_id + "/depth/"
    pose_dir = params["shapenet_renderings"] + synset_id + "/" + model_id + "/pose/"
    occ_file = params["shapenet_voxelized"] + synset_id + "/" + model_id + "__0__.txt"

    grid = raymarch_n_views(color_img_dir, depth_img_dir, pose_dir, occ_file)

    txt_file = params["network_output"] + "vox.txt"
    ply_file = params["network_output"] + "vox.ply"
    save_vox_as_txt(txt_file, grid)
    voxel_grid.txt_to_mesh(txt_file, ply_file)


if __name__ == '__main__':
    synset_id = '03001627'
    model_id = '1a6f615e8b1b5ae4dbbc9440457e303e'

    voxelize(synset_id, model_id)


