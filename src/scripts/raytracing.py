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
import perspective_projection as projection


# Debugging helper function : saves the color values with corresponding pixel coordinates
# Considers image to have white background color : so we dont save those values
def save_image_info_txt(image, file):
    with open(file, "w") as f:
        for u in range(image.shape[0]):
            for v in range(image.shape[1]):
                if not np.all(image[v, u, :] == 255):
                    f.write("%d %d %d %d %d\n" % (u, v, image[v,u,0], image[v,u,1], image[v,u,2]))


def raycast(txt_file, pose_file, color_file, depth_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im = Image.open(color_file)
    color = torch.from_numpy(np.array(im)).float().to(device)

    depth_im = Image.open(depth_file)
    depth = torch.from_numpy(np.array(depth_im)).float().to(device)

    pose = loader.load_pose(pose_file).to(device)
    projection_helper = projection.ProjectionHelper()
    index_map = projection_helper.compute_index_mapping(pose).to(device)

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

    depth_im = Image.open(depth_file)
    depth = torch.from_numpy(np.array(depth_im)).float().to(device)

    pose = loader.load_pose(pose_file).to(device)
    projection_helper = projection.ProjectionHelper()
    index_map = projection_helper.compute_index_mapping(pose).to(device)

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

    outdir = params["shapenet_raytraced"] + synset_id
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    pose_file = params["shapenet_renderings"] + synset_id + "/" + model_id + "/pose/pose00.json"
    image_file = params["shapenet_renderings"] + synset_id + "/" + model_id + "/color/color00.png"
    depth_file = params["shapenet_renderings"] + synset_id + "/" + model_id + "/depth/depth00.png"
    voxel_file = params["network_output"] + "/" + model_id + ".ply"
    voxel_txt_file = params["network_output"] + "/" + model_id + ".txt"

    if os.path.exists(voxel_file) and os.path.exists(voxel_txt_file):
        print(synset_id, " : ", model_id, " already exists. Skipping......")
        return

    raycast_depth(voxel_txt_file, pose_file, depth_file)

    txt_to_mesh(voxel_txt_file, voxel_file)


if __name__ == '__main__':
    synset_lst = []
    synset_lst.append("02747177")

    params = JSONHelper.read("./parameters.json")

    failed_cases = {}
    file = params["shapenet_raytraced"] + "failed_cases.json"

    synset_id = "02747177"
    model_id = "501154f25599ee80cb2a965e75be701c"
    generate_raycasted_model(synset_id, model_id)


    # for synset_id in synset_lst:
    #     for f in glob.glob(params["shapenet"] + synset_id + "/*/models/model_normalized.obj"):
    #         model_id = f.split("/", 8)[6]
    #         print(synset_id, " : ", model_id)
    #
    #         try:
    #             if not synset_id in failed_cases.keys():
    #                 failed_cases[synset_id] = []
    #
    #             generate_raycasted_model(synset_id, model_id)
    #         except:
    #             failed_cases[synset_id].append(model_id)
    #             pass
    #
    #     print("Finished raycasting synset ", synset_id)
    #
    # JSONHelper.write(file, failed_cases)
