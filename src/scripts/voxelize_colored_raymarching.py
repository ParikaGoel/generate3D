import sys
sys.path.append('./Network')
import os
import glob
import pathlib
import JSONHelper
import numpy as np
from camera import *
from PIL import Image
from voxel_grid import *
import dataset_loader as loader

# this will not work if the nearest plane for the camera is farther than the unit depth
# Fix would be to calculate the intersection of each ray with the nearest plane and then iterate from there
# ToDo : Implement the intersection logic
def raycast(voxel_grid, occ_grid, cam, color_file, depth_file):
    im = Image.open(color_file)
    color = np.array(im)

    depth_im = Image.open(depth_file)
    depth = np.array(depth_im)

    width = color.shape[1]
    height = color.shape[0]
    cx = cam.center[0]
    cy = cam.center[1]
    focal = cam.focal[0]

    # Might have to start at z = -0.5
    raycast_depth = cam.z_near

    for u in range(width):
        for v in range(height):
            done = False
            x = (u - cx) * raycast_depth / focal
            y = (v - cy) * raycast_depth / focal
            ray = np.array([x, -y, -raycast_depth])
            ray_length = np.sqrt(np.sum(ray ** 2))
            unit_ray = ray / ray_length

            while voxel_grid.contains_global_coord(ray) and not done:
                grid_coord = voxel_grid.get_grid_coord(ray)
                if grid_coord is not None and occ_grid[grid_coord[0], grid_coord[1], grid_coord[2]] == 1:
                    if not np.all(depth[v, u] == 0):
                        voxel_grid.set_occupancy(grid_coord, 1)
                        voxel_grid.set_color(grid_coord, color[v, u])
                        done = True

                ray_length = ray_length + (voxel_grid.voxel_scale / 2)
                ray = unit_ray * ray_length


def generate_raycasted_model(synset_id, model_id):
    params = JSONHelper.read("./parameters.json")

    outdir = params["network_output"] + synset_id
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    cam_file = params["shapenet_renderings"] + synset_id + "/" + model_id + "/pose/pose00.json"
    image_file = params["shapenet_renderings"] + synset_id + "/" + model_id + "/color/color00.png"
    depth_file = params["shapenet_renderings"] + synset_id + "/" + model_id + "/depth/depth00.png"
    occ_file = params["shapenet_voxelized"] + synset_id + "/" + model_id + "__0__.txt"

    occ_grid = loader.load_sample(occ_file)[0]
    voxel_file = outdir + "/" + model_id + ".ply"
    voxel_txt_file = outdir + "/" + model_id + ".txt"

    if os.path.exists(voxel_file) and os.path.exists(voxel_txt_file):
        print(synset_id, " : ", model_id, " already exists. Skipping......")
        return

    cam = load_camera(cam_file)
    voxel_grid = create_voxel_grid(cam)
    raycast(voxel_grid, occ_grid, cam, image_file, depth_file)

    voxel_grid.save_vox(voxel_txt_file)

    txt_to_mesh(voxel_txt_file, voxel_file, grid_size=abs(cam.z_far - cam.z_near))


if __name__ == '__main__':
    synset_lst = []
    synset_lst.append("03001627")

    params = JSONHelper.read("./parameters.json")

    failed_cases = {}
    file = params["shapenet_raytraced"] + "failed_cases.json"

    for synset_id in synset_lst:
        for f in glob.glob(params["shapenet"] + synset_id + "/*/models/model_normalized.obj"):
            model_id = f.split("/", 11)[8]
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
