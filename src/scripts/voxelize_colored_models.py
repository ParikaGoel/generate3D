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
    intrinsic[0][2] = config.img_width / 2
    intrinsic[1][2] = config.img_height / 2
    return intrinsic

def make_world_to_grid():
    world_to_grid = torch.eye(4)
    world_to_grid[0][3] = world_to_grid[1][3] = world_to_grid[2][3] = 0.5
    world_to_grid *= config.vox_dim
    world_to_grid[3][3] = 1.0
    return world_to_grid


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


def compute_index_map(device, occ_grid, world_to_camera):
    index_map = torch.empty((config.img_height, config.img_width), dtype=int).fill_(-1).to(device)
    grid_to_world = torch.inverse(make_world_to_grid()).to(device)
    intrinsic = make_intrinsic().to(device)

    occ = torch.flatten(occ_grid, start_dim=0, end_dim=-1)
    occ_mask = torch.eq(occ, 1)

    lin_ind_volume = torch.arange(0, config.vox_dim * config.vox_dim * config.vox_dim,
                                  out=torch.LongTensor()).to(device)
    lin_ind_volume = lin_ind_volume[occ_mask]
    grid_coords = grid_to_world.new(4, lin_ind_volume.size(0))
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
    vertex_coords = torch.transpose(vertex_coords, 0, 1)

    corners = torch.stack([grid_coords.int() + vertex_coords[i][:, None] for i in range(8)])

    # transform to current frame
    camera_coords = torch.matmul(world_to_camera, torch.matmul(grid_to_world, corners.float()))
    depth = camera_coords.new(camera_coords.size(2) + 1)
    depth[:-1] = torch.min(torch.abs(camera_coords[:, 2, :]), dim=0).values
    depth[-1] = torch.max(depth[:-1]) + 1

    # project into image
    p = camera_coords.clone()
    p[:, 1, :] = -p[:, 1, :]  # perspective projection flips the image in y- direction
    p[:, 0] = (p[:, 0] * intrinsic[0][0]) / torch.abs(p[:, 2]) + intrinsic[0][2]
    p[:, 1] = (p[:, 1] * intrinsic[1][1]) / torch.abs(p[:, 2]) + intrinsic[1][2]
    p = torch.round(p).long()[:, 0:2, :]

    p = torch.clamp(p, min=0, max=511)
    pmin = torch.min(p, dim=0).values
    pmax = torch.max(p, dim=0).values

    index_map = index_map.fill_(lin_ind_volume.size(0))

    for i in range(lin_ind_volume.size(0)):
        vals = index_map[pmin[1, i]:pmax[1, i], pmin[0, i]:pmax[0, i]]
        vals = torch.flatten(vals)
        depth_vals = torch.index_select(depth, 0, vals)
        mask = torch.gt(depth_vals, depth[i])
        vals[mask] = i
        vals = vals.reshape((pmax[1, i] - pmin[1, i], pmax[0, i] - pmin[0, i]))
        index_map[pmin[1, i]:pmax[1, i], pmin[0, i]:pmax[0, i]] = vals

    lin_index_map = torch.flatten(index_map, start_dim=0, end_dim=-1)
    invalid_mask = torch.eq(lin_index_map, lin_ind_volume.size(0))
    lin_index_map[invalid_mask] = 0
    lin_index_map = torch.index_select(lin_ind_volume, 0, lin_index_map)
    lin_index_map[invalid_mask] = -1

    return lin_index_map


def raymarch(device, color_grid, occ_grid, color_file, depth_file, pose_file):
    pose = loader.load_pose(pose_file).to(device)
    img = loader.load_img(color_file, color=True).to(device)
    img = torch.reshape(img, (-1, 3)).float()

    depth_im = Image.open(depth_file)
    depth = torch.flatten(torch.from_numpy(np.array(depth_im))).float()

    invalid_depth = torch.eq(depth, 0)
    img[invalid_depth] = 0

    index_map = compute_index_map(device, occ_grid, pose).long().to(device)

    grid = torch.empty(((config.vox_dim * config.vox_dim * config.vox_dim) + 1, 3),
                       dtype=torch.float).fill_(0).to(device)

    invalid_mask = torch.eq(index_map, -1)
    index_map[invalid_mask] = grid.size(0) - 1

    indices, indices_count = torch.unique(index_map, return_counts=True)
    grid.index_add_(0, index_map, img)
    grid[indices] = grid[indices] / indices_count[:, None]
    grid[indices] = torch.clamp(grid[indices], max=255)

    color_grid[indices] = grid[indices]

    return color_grid


def raymarch_n_views(device, color_img_dir, depth_dir, pose_dir, occ_file):
    img_files = sorted(os.listdir(color_img_dir))
    depth_files = sorted(os.listdir(depth_dir))
    pose_files = sorted(os.listdir(pose_dir))
    occ_grid = loader.load_sample(occ_file)[0]

    color_grid = torch.empty(((config.vox_dim * config.vox_dim * config.vox_dim)+1, 3),
                             dtype=torch.float).fill_(256).to(device)

    for i in range(len(img_files)):
        color_grid = raymarch(device, color_grid, occ_grid, color_img_dir + img_files[i], depth_dir + depth_files[i], pose_dir + pose_files[i])

    color_grid = color_grid[:-1, :]
    return color_grid


def voxelize(device, synset_id, model_id):
    color_img_dir = params["shapenet_renderings"] + synset_id + "/" + model_id + "/test/color/"
    depth_img_dir = params["shapenet_renderings"] + synset_id + "/" + model_id + "/test/depth/"
    pose_dir = params["shapenet_renderings"] + synset_id + "/" + model_id + "/test/pose/"
    occ_file = params["shapenet_voxelized"] + synset_id + "/" + model_id + "__0__.txt"

    color_grid = raymarch_n_views(device, color_img_dir, depth_img_dir, pose_dir, occ_file)

    txt_file = params["network_output"] + "vox.txt"
    ply_file = params["network_output"] + "vox.ply"
    save_vox_as_txt(txt_file, color_grid)
    voxel_grid.txt_to_mesh(txt_file, ply_file)


if __name__ == '__main__':
    # synset_id = '03001627'
    # model_id = '1a6f615e8b1b5ae4dbbc9440457e303e'

    synset_id = '04379243'
    model_id = '1a00aa6b75362cc5b324368d54a7416f'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    voxelize(device, synset_id, model_id)


