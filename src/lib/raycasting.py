from camera import *
from voxel_grid import *
from image import *
from PIL import Image
import numpy as np


# In ideal pinhole camera, fx and fy are same. We are making that assumption here
def create_voxel_grid(cam):
    focal_length = cam.focal
    principal_point = cam.center
    voxel_min_bound = np.array([-principal_point[0], -principal_point[1], cam.z_near])
    grid_size = abs(cam.z_far - cam.z_near) #cam.resolution[0] * 2
    voxel_dim = 32
    voxel_grid = VoxelGrid(voxel_min_bound, voxel_dim, grid_size)
    return voxel_grid


def raycast(voxel_grid, cam, image):
    width = image.shape[0]
    height = image.shape[1]

    f = cam.focal[0]
    cx = cam.center[0]
    cy = cam.center[1]

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
                    if image[u, v, 0] == 255 and image[u, v, 1] == 255 and image[u, v, 2] == 255:
                        voxel_grid.set_occupancy(grid_coord, 0)
                    else:
                        voxel_grid.set_occupancy(grid_coord, 1)
                        voxel_grid.set_color(grid_coord, image[u,v])
                ray_length = ray_length + voxel_grid.get_voxel_scale()
                ray = normalized_ray * ray_length


if __name__ == '__main__':
    cam_file = 'results/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/renderings/camera/cam1.json'
    image_file = 'results/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/renderings/color/color1.png'
    voxel_file = 'results/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/raycasted_voxel/voxel1.ply'
    cam = load_camera(cam_file)
    # image = load_image(image_file)
    im = Image.open(image_file)
    np_img = np.array(im)
    np_img = np_img[:, :, 0:3]
    voxel_grid = create_voxel_grid(cam)
    raycast(voxel_grid, cam, np_img)

    # Since we have done raycasting in camera system, apply the cam to world transform while saving the voxel
    voxel_grid.save_as_ply(voxel_file, cam.pose)


