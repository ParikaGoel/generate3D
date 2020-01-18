import os
import glob
import pathlib
import trimesh
import JSONHelper
import numpy as np
from camera import *
from PIL import Image
from voxel_grid import *
import matplotlib.pyplot as plt


# Debugging helper function : saves the color values with corresponding pixel coordinates
# Considers image to have white background color : so we dont save those values
def save_image_info_txt(image, file):
    with open(file, "w") as f:
        for u in range(image.shape[0]):
            for v in range(image.shape[1]):
                if not np.all(image[v, u, :] == 255):
                    f.write("%d %d %d %d %d\n" % (u, v, image[v,u,0], image[v,u,1], image[v,u,2]))


# this will not work if the nearest plane for the camera is farther than the unit depth
# Fix would be to calculate the intersection of each ray with the nearest plane and then iterate from there
# ToDo : Implement the intersection logic
def raycast(voxel_grid, cam, color_file, depth_file):
    im = Image.open(color_file)
    color = np.array(im)

    depth_im = Image.open(depth_file)
    depth = np.array(depth_im)

    width = color.shape[1]
    height = color.shape[0]
    cx = cam.center[0]
    cy = cam.center[1]
    focal = cam.focal[0]

    raycast_depth = cam.z_near

    for u in range(width):
        for v in range(height):
            x = (u - cx) * raycast_depth / focal
            y = (v - cy) * raycast_depth / focal
            # ray = np.array([x, -y, -1])
            ray = np.array([x, -y, -raycast_depth])
            ray_length = np.sqrt(np.sum(ray ** 2))
            unit_ray = ray / ray_length

            # # test projecting only one layer of the image
            # grid_coord = voxel_grid.get_grid_coord(ray)
            # if grid_coord is not None:
            #     if not np.all(image[v, u] > 200):
            #         voxel_grid.set_occupancy(grid_coord, 1)
            #         voxel_grid.set_color(grid_coord, image[v, u])

            while voxel_grid.contains_global_coord(ray):
                grid_coord = voxel_grid.get_grid_coord(ray)
                if grid_coord is not None:
                    if not np.all(depth[v, u] == 0):
                        voxel_grid.set_occupancy(grid_coord, 1)
                        voxel_grid.set_color(grid_coord, color[v, u])

                ray_length = ray_length + (voxel_grid.voxel_scale / 2)
                ray = unit_ray * ray_length


def generate_raycasted_model(synset_id, model_id):
    params = JSONHelper.read("./parameters.json")

    outdir = params["shapenet_raytraced"] + synset_id
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    cam_file = params["shapenet_renderings"] + synset_id + "/" + model_id + "/cam/cam0.json"
    image_file = params["shapenet_renderings"] + synset_id + "/" + model_id + "/color/color0.png"
    depth_file = params["shapenet_renderings"] + synset_id + "/" + model_id + "/depth/depth0.png"
    voxel_file = outdir + "/" + model_id + ".ply"
    voxel_txt_file = outdir + "/" + model_id + ".txt"

    if os.path.exists(voxel_file) and os.path.exists(voxel_txt_file):
        print(synset_id, " : ", model_id, " already exists. Skipping......")
        return

    cam = load_camera(cam_file)
    voxel_grid = create_voxel_grid(cam)
    raycast(voxel_grid, cam, image_file, depth_file)

    voxel_grid.save_vox(voxel_txt_file)

    txt_to_mesh(voxel_txt_file, voxel_file, grid_size=abs(cam.z_far - cam.z_near))


# if __name__ == '__main__':
#     train_list = []
#     train_list.append({'synset_id': '02933112', 'model_id': '2f0fd2a5e181b82a4267f85fb94fa2e7'})
#     train_list.append({'synset_id': '02933112', 'model_id': 'a46d947577ecb54a6bdcd672c2b17215'})
#     train_list.append({'synset_id': '02933112', 'model_id': '37ba0371250bcd6de117ecc943aca233'})
#     train_list.append({'synset_id': '02933112', 'model_id': 'bd2bcee265b1ee1c7c373e0e7470a338'})
#     train_list.append({'synset_id': '02933112', 'model_id': '8a2aadf8fc4f092c5ee1a94f1da3a5e'})
#
#     train_list.append({'synset_id': '02942699', 'model_id': '6d036fd1c70e5a5849493d905c02fa86'})
#     train_list.append({'synset_id': '02942699', 'model_id': '97690c4db20227d248e23e2c398d8046'})
#     train_list.append({'synset_id': '02942699', 'model_id': 'e9e22de9e4c3c3c92a60bd875e075589'})
#     train_list.append({'synset_id': '02942699', 'model_id': '51176ec8f251800165a1ced01089a2d6'})
#     train_list.append({'synset_id': '02942699', 'model_id': '46c09085e451de8fc3c192db90697d8c'})
#
#     train_list.append({'synset_id': '02946921', 'model_id': 'ebcbb82d158d68441f4c1c50f6e9b74e'})
#     train_list.append({'synset_id': '02946921', 'model_id': '3703ada8cc31df4337b00c4c2fbe82aa'})
#     train_list.append({'synset_id': '02946921', 'model_id': 'fd40fa8939f5f832ae1aa888dd691e79'})
#     train_list.append({'synset_id': '02946921', 'model_id': '3fd8dae962fa3cc726df885e47f82f16'})
#     train_list.append({'synset_id': '02946921', 'model_id': 'b1980d6743b7a98c12a47018402419a2'})
#
#     train_list.append({'synset_id': '03636649', 'model_id': 'bde9b62e181cd4694fb315ce917a9ec2'})
#     train_list.append({'synset_id': '03636649', 'model_id': '967b6aa33d17c109e81edb73cdd34eeb'})
#     train_list.append({'synset_id': '03636649', 'model_id': '6ffb0636180aa5d78570a59d0416a26d'})
#     train_list.append({'synset_id': '03636649', 'model_id': 'f449dd0eb25773925077539b37310c29'})
#     train_list.append({'synset_id': '03636649', 'model_id': '989694b21ed5752d4c61a7cce317bfb7'})
#
#     for data in train_list:
#         synset_id = data['synset_id']
#         model_id = data['model_id']
#         generate_raycasted_model(synset_id, model_id)


# if __name__ == '__main__':
#     synset_id = "02747177"
#     model_id = "85d8a1ad55fa646878725384d6baf445"
#
#     generate_raycasted_model(synset_id, model_id)


# if __name__ == '__main__':
#     params = JSONHelper.read("./parameters.json")
# 
#     for f in glob.glob(params["shapenet_renderings"] + "/**/*/color/color0.png"):
#         synset_id = f.split("/", 6)[4]
#         model_id = f.split("/", 6)[5]
# 
#         generate_raycasted_model(synset_id, model_id)

if __name__ == '__main__':
    synset_lst = []
    synset_lst.append("04330267")
    # synset_lst.append("02801938")
    # synset_lst.append("02843684")
    # synset_lst.append("02880940")
    # synset_lst.append("02954340")
    # synset_lst.append("03085013")

    params = JSONHelper.read("./parameters.json")
    
    for synset_id in synset_lst:
        for f in glob.glob(params["shapenet_renderings"] + synset_id + "/*/color/color0.png"):
            print(f,'\n')
            model_id = f.split("/", 10)[8]
            print(synset_id, " ", model_id)
            generate_raycasted_model(synset_id, model_id)
        print("Finished raycasting synset ", synset_id)
