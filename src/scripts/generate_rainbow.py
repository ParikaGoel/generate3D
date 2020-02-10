import numpy as np
from camera import *
from voxel_grid import *


def generate_rainbow_colors(count):

   red = np.sin(0.3*count + 0) * 127 + 128
   grn = np.sin(0.3*count + 2) * 127 + 128
   blu = np.sin(0.3*count + 4) * 127 + 128

   color = np.array((red.astype(int), grn.astype(int), blu.astype(int)))
   return color


def update_color(color):
    if color[0] < 255:
        color[0] += 5
    elif color[1] < 255:
        color[1] += 5
    elif color[2] < 255:
        color[2] += 5

    return color


def raycast(voxel_grid):
    width = height = 512
    cx = cy = 512 / 2
    focal = 525.0

    raycast_depth = 0.5

    count = 0

    for u in range(width):
        for v in range(height):
            x = (u - cx) * raycast_depth / focal
            y = (v - cy) * raycast_depth / focal
            color = generate_rainbow_colors(count)
            ray = np.array([x, -y, -raycast_depth])
            ray_length = np.sqrt(np.sum(ray ** 2))
            unit_ray = ray / ray_length

            # test projecting only one layer of the image
            # grid_coord = voxel_grid.get_grid_coord(ray)
            # if grid_coord is not None:
            #     voxel_grid.set_occupancy(grid_coord, 1)
            #     voxel_grid.set_color(grid_coord, color)

            while voxel_grid.contains_global_coord(ray):
                grid_coord = voxel_grid.get_grid_coord(ray)
                if grid_coord is not None:
                        voxel_grid.set_occupancy(grid_coord, 1)
                        voxel_grid.set_color(grid_coord, color)

                ray_length = ray_length + (voxel_grid.voxel_scale / 2)
                ray = unit_ray * ray_length

            count = count + 1


if __name__ == '__main__':
    cam_file = "/media/sda2/shapenet/test/fd013bea1e1ffb27c31c70b1ddc95e3f/pose/pose0.json"
    image_file = "/media/sda2/shapenet/test/fd013bea1e1ffb27c31c70b1ddc95e3f/color/color0.png"
    voxel_file = "/media/sda2/shapenet/test/rainbow_0_.ply"
    voxel_txt_file = "/media/sda2/shapenet/test/rainbow_0_.txt"

    cam = load_camera(cam_file)
    voxel_grid = create_voxel_grid(cam)
    raycast(voxel_grid)

    voxel_grid.save_vox(voxel_txt_file)

    txt_to_mesh(voxel_txt_file, voxel_file, grid_size=abs(cam.z_far - cam.z_near))