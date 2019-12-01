from camera import *
from voxel_grid import *
from image import *
import numpy as np


# In ideal pinhole camera, fx and fy are same. We are making that assumption here
def create_voxel_grid(cam):
    focal_length = cam.get_fx()
    principal_point_x = cam.get_cx()
    principal_point_y = cam.get_cy()

    voxel_min_bound = np.array([-2*principal_point_x, -2*principal_point_y, focal_length])
    grid_size = cam.get_width() * 2
    voxel_dim = 32
    voxel_grid = VoxelGrid(voxel_min_bound, voxel_dim, grid_size)
    return voxel_grid


def raycast(voxel_grid, cam, image):
    width = image.shape[0]
    height = image.shape[1]

    f = cam.get_fx()
    cx = cam.get_cx()
    cy = cam.get_cy()

    for u in range(width):
        for v in range(height):
            intersection_point = np.array([u-cx, v-cy, f])

            # lets start the ray from the center of the first voxel it hits
            ray = intersection_point + (voxel_grid.get_voxel_scale() / 2)
            ray_length = np.sqrt(np.sum(ray**2))
            normalized_ray = ray / ray_length

            while voxel_grid.contains_global_coord(ray):
                grid_coord = voxel_grid.get_grid_coord(ray)
                if grid_coord is not None:
                    if image[u, v, 0] == 1.0 and image[u, v, 1] == 1.0 and image[u, v, 2] == 1.0:
                        voxel_grid.set_occupancy(grid_coord, 0)
                    else:
                        voxel_grid.set_occupancy(grid_coord, 1)
                ray_length = ray_length + voxel_grid.get_voxel_scale()
                ray = normalized_ray * ray_length


if __name__ == '__main__':
    cam_file = 'results/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/camera/20191201230817.json'
    image_file = 'results/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/image/20191201230817.png'
    voxel_file = 'results/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/raycasted_voxel/20191201230817.ply'
    cam = load_camera(cam_file)
    image = load_image(image_file)
    voxel_grid = create_voxel_grid(cam)
    raycast(voxel_grid, cam, image)
    voxel_grid.save_as_ply(voxel_file)


