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
def raycast(voxel_grid, cam, image):
    width = image.shape[1]
    height = image.shape[0]
    cx = cam.center[0]
    cy = cam.center[1]
    focal = cam.focal[0]

    depth = cam.z_near

    for u in range(width):
        for v in range(height):
            x = (u - cx) * depth / focal
            y = (v - cy) * depth / focal
            # ray = np.array([x, -y, -1])
            ray = np.array([x, -y, -depth])
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
                    if not np.all(image[v, u] > 200):
                        voxel_grid.set_occupancy(grid_coord, 1)
                        voxel_grid.set_color(grid_coord, image[v, u])

                ray_length = ray_length + (voxel_grid.voxel_scale / 2)
                ray = unit_ray * ray_length


if __name__ == '__main__':
    # stool
    catid = "04379243"
    id = "142060f848466cad97ef9a13efb5e3f7"

    params = JSONHelper.read("./parameters.json")

    cam_file = params["shapenet_renderings"] + catid + "/" + id + "/cam/cam0.json"
    image_file = params["shapenet_renderings"] + catid + "/" + id + "/color/color0.png"
    voxel_file = params["shapenet_raytraced"] + catid + "/" + id + ".ply"
    voxel_txt_file = params["shapenet_raytraced"] + catid + "/" + id + ".txt"

    im = Image.open(image_file)
    np_img = np.array(im)
    np_img = np_img[:, :, 0:3]

    # save_image_info_txt(np_img, img_txt_file)

    cam = load_camera(cam_file)
    voxel_grid = create_voxel_grid(cam)
    raycast(voxel_grid, cam, np_img)

    voxel_grid.save_vox(voxel_txt_file)

    # # Since we have done raycasting in camera system, apply the cam to world transform while saving the voxel
    voxel_grid.to_mesh(voxel_file, cam.pose)
