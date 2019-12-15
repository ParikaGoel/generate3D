import json
import trimesh
import numpy as np
from camera import *
from PIL import Image
from voxel_grid import *


def load_camera(cam_file):
    with open(cam_file) as camera_file:
        data = json.load(camera_file)

        # read the intrinsic parameters
        resolution = [data['intrinsic']['width'], data['intrinsic']['height']]
        focal = [data['intrinsic']['fx'], data['intrinsic']['fy']]
        znear = data['intrinsic']['z_near']
        zfar = data['intrinsic']['z_far']

        cam = trimesh.scene.Camera(resolution=resolution, focal=focal, z_near=znear, z_far=zfar)

        # read the camera pose
        pose = np.empty(0)
        for val in data['pose']:
            pose = np.append(pose, val)
        pose = np.transpose(np.reshape(pose, (4, 4)))

        return cam, pose


def create_voxel_grid(cam):
    grid_size = abs(cam.z_far - cam.z_near)
    voxel_min_bound = np.array([-grid_size/2, -grid_size/2, -cam.z_near])
    voxel_dim = 32
    voxel_grid = VoxelGrid(voxel_min_bound, voxel_dim, grid_size)
    print("Min bound: ", voxel_grid.get_min_bound())
    print("Max bound: ", voxel_grid.get_max_bound())
    print("Voxel Scale: ", voxel_grid.get_voxel_scale())
    return voxel_grid


def raycast(voxel_grid, cam, image):
    xy, pixels = trimesh.scene.cameras.ray_pixel_coords(cam)
    rays = np.column_stack((xy, -np.ones_like(xy[:, :1])))

    for count in range(pixels.shape[0]):
        pixel = pixels[count]
        u = 511 if pixel[0] == 512 else pixel[0]
        v = 511 if pixel[1] == 512 else pixel[1]

        ray = rays[count]
        ray_length = np.sqrt(np.sum(ray**2))
        unit_ray = ray / ray_length

        while voxel_grid.contains_global_coord(ray):
            grid_coord = voxel_grid.get_grid_coord(ray)
            if grid_coord is not None:
                if np.all(image[v,u] == 255):
                    voxel_grid.set_occupancy(grid_coord, 0)
                else:
                    voxel_grid.set_occupancy(grid_coord, 1)
                    voxel_grid.set_color(grid_coord, image[v, u])
            ray_length = ray_length + (voxel_grid.get_voxel_scale()/2)
            ray = unit_ray * ray_length


if __name__ == '__main__':
    cam_file = 'results/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/renderings/camera/cam1.json'
    image_file = 'results/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/renderings/color/color1.png'
    voxel_file = 'results/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/raycasted_voxel/voxel1_cam.ply'
    im = Image.open(image_file)
    np_img = np.array(im)
    np_img = np_img[:, :, 0:3]

    # for u in range(512):
    #     for v in range(512):
    #         if not np.all(np_img[v,u,:] == 255):
    #             print(u, v, np_img[v,u,:])



    cam, cam_pose = load_camera(cam_file)
    voxel_grid = create_voxel_grid(cam)
    raycast(voxel_grid, cam, np_img)

    # Since we have done raycasting in camera system, apply the cam to world transform while saving the voxel
    voxel_grid.save_as_ply(voxel_file, cam_pose)
