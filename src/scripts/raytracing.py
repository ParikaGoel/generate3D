import sys
sys.path.append('./Network')
import os
import glob
import torch
import config
import pathlib
import trimesh
import JSONHelper
import numpy as np
from camera import *
from PIL import Image
from voxel_grid import *
import dataset_loader as loader


# Debugging helper function : saves the color values with corresponding pixel coordinates
# Considers image to have white background color : so we dont save those values
def save_image_info_txt(image, file):
    with open(file, "w") as f:
        for u in range(image.shape[0]):
            for v in range(image.shape[1]):
                if not np.all(image[v, u, :] == 255):
                    f.write("%d %d %d %d %d\n" % (u, v, image[v,u,0], image[v,u,1], image[v,u,2]))


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


def raycast(txt_file, pose_file, color_file, depth_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_to_world = torch.inverse(make_world_to_grid()).to(device)

    im = Image.open(color_file)
    color = torch.from_numpy(np.array(im)).float().to(device)

    depth_im = Image.open(depth_file)
    depth = torch.from_numpy(np.array(depth_im)).float().to(device)

    pose = loader.load_pose(pose_file).to(device)
    index_map = compute_index_mapping(pose, grid_to_world, device).to(device)

    voxel_grid = torch.empty((config.vox_dim * config.vox_dim * config.vox_dim, 3),
                         dtype=torch.float).fill_(255).to(device)

    index_map = index_map.reshape((4, -1))

    for i in range(voxel_grid.size(0)):
        voxel_grid[i] = torch.mean(torch.flatten(color[index_map[1, i]:index_map[3, i], index_map[0, i]:index_map[2, i]], start_dim=0, end_dim=1), dim=0)

    voxel_grid[torch.isnan(voxel_grid)] = 255
    voxel_grid = torch.ceil(voxel_grid).cpu().numpy().astype(np.uint8)
    voxel_grid = voxel_grid.reshape((config.vox_dim, config.vox_dim, config.vox_dim, 3)).transpose(2, 1, 0, 3)

    positions = np.argwhere(np.all(voxel_grid < 255, axis=3))

    with open(txt_file, "w") as f:
        for i, j, k in positions:
            color = voxel_grid[i, j, k]
            data = np.column_stack((i, j, k, color[0], color[1], color[2]))
            np.savetxt(f, data, fmt='%d %d %d %d %d %d', delimiter=' ')

def raycast_depth(txt_file, pose_file, depth_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_to_world = torch.inverse(make_world_to_grid()).to(device)

    depth_im = Image.open(depth_file)
    depth = torch.from_numpy(np.array(depth_im)).float().to(device)

    pose = loader.load_pose(pose_file).to(device)
    index_map = compute_index_mapping(pose, grid_to_world, device).to(device)

    index_map = index_map.reshape((4, -1))

    voxel_grid = torch.stack([torch.tensor(1 if np.any(depth[index_map[1, i]:index_map[3, i], index_map[0, i]:index_map[2, i]].cpu().numpy() > 0) else 0)
                          for i in range(32768)]).reshape(32, 32, 32)
    voxel_grid = torch.transpose(voxel_grid, 0, 2)

    positions = np.where(voxel_grid == 1)

    with open(txt_file, "w") as f:
        for i, j, k in zip(*positions):
            color = np.array([169, 0, 255])
            data = np.column_stack((i, j, k, color[0], color[1], color[2]))
            np.savetxt(f, data, fmt='%d %d %d %d %d %d', delimiter=' ')


def generate_raycasted_model(synset_id, model_id):
    params = JSONHelper.read("./parameters.json")

    outdir = "/media/sda2/shapenet/shapenet-raytraced/" + synset_id
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    pose_file = "/media/sda2/shapenet/shapenet-renderings/" + synset_id + "/" + model_id + "/pose/pose00.json"
    image_file = "/media/sda2/shapenet/shapenet-renderings/" + synset_id + "/" + model_id + "/color/color00.png"
    depth_file = "/media/sda2/shapenet/shapenet-renderings/" + synset_id + "/" + model_id + "/depth/depth00.png"
    voxel_file = outdir + "/" + model_id + ".ply"
    voxel_txt_file = outdir + "/" + model_id + ".txt"

    if os.path.exists(voxel_file) and os.path.exists(voxel_txt_file):
        print(synset_id, " : ", model_id, " already exists. Skipping......")
        return

    raycast_depth(voxel_txt_file, pose_file, depth_file)

    txt_to_mesh(voxel_txt_file, voxel_file)


if __name__ == '__main__':
    synset_lst = []
    synset_lst.append("03001627")

    params = JSONHelper.read("./parameters.json")

    failed_cases = {}
    file = "/media/sda2/shapenet/shapenet-raytraced/failed_cases.json"


    for synset_id in synset_lst:
        for f in glob.glob("/media/sda2/shapenet/shapenet-renderings/" + synset_id + "/*/color/color00.png"):
            model_id = f.split("/", 10)[6]
            print(synset_id, " : ", model_id)

            try:
                if not synset_id in failed_cases.keys():
                    failed_cases[synset_id] = []

                generate_raycasted_model(synset_id, model_id)
            except:
                failed_cases[synset_id].append(model_id)
                pass

        print("Finished raycasting synset ", synset_id)

    JSONHelper.write(file, failed_cases)
