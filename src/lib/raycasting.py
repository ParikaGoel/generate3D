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
    voxel_min_bound = np.array([-grid_size / 2, -grid_size / 2, -cam.z_near])
    voxel_dim = 32
    voxel_grid = VoxelGrid(voxel_min_bound, voxel_dim, grid_size)
    print("Min bound: ", voxel_grid.get_min_bound())
    print("Max bound: ", voxel_grid.get_max_bound())
    print("Voxel Scale: ", voxel_grid.get_voxel_scale())
    return voxel_grid


# this will not work if the nearest plane for the camera is farther than the unit depth
# Fix would be to calculate the intersection of each ray with the nearest plane and then iterate from there
# ToDo : Implement the intersection logic
def raycast(voxel_grid, cam, image):
    width = image.shape[1]
    height = image.shape[0]
    center = width / 2
    focal = cam.focal[0]

    nVoxels = 0
    # voxel_txt_file = 'results/02747177/85d8a1ad55fa646878725384d6baf445/text/voxel.txt'
    # voxel_txt_file = 'results/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/text/voxel.txt'
    voxel_txt_file = 'results/03001627/bdc892547cceb2ef34dedfee80b7006/text/voxel.txt'
    voxel_file = open(voxel_txt_file, "w")
    for u in range(width):
        for v in range(height):
            x = (u - center) / focal
            y = (v - center) / focal
            ray = np.array([x, -y, -1])
            ray_length = np.sqrt(np.sum(ray ** 2))
            unit_ray = ray / ray_length

            while voxel_grid.contains_global_coord(ray):
                grid_coord = voxel_grid.get_grid_coord(ray)
                if grid_coord is not None:
                    if np.all(image[v, u] == 255):
                        voxel_grid.set_occupancy(grid_coord, 0)
                    else:
                        nVoxels += 1
                        voxel_grid.set_occupancy(grid_coord, 1)
                        voxel_grid.set_color(grid_coord, image[v, u])
                        voxel_file.write("%f %f %f %d %d %d %d %d %d\n" % (
                            ray[0], ray[1], ray[2], grid_coord[0], grid_coord[1], grid_coord[2], image[v, u, 0],
                            image[v, u, 1], image[v, u, 2]))

                ray_length = ray_length + (voxel_grid.get_voxel_scale() / 2)
                ray = unit_ray * ray_length


if __name__ == '__main__':
    # trashbin
    # cam_file = 'results/02747177/85d8a1ad55fa646878725384d6baf445/renderings/camera/cam1.json'
    # image_file = 'results/02747177/85d8a1ad55fa646878725384d6baf445/renderings/color/color1.png'
    # voxel_file = 'results/02747177/85d8a1ad55fa646878725384d6baf445/raycasted_voxel/voxel.ply'

    # bench
    # cam_file = 'results/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/renderings/camera/cam1.json'
    # image_file = 'results/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/renderings/color/color1.png'
    # voxel_file = 'results/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/raycasted_voxel/voxel.ply'

    # chair
    # cam_file = 'results/03001627/bdc892547cceb2ef34dedfee80b7006/renderings/camera/cam1.json'
    # image_file = 'results/03001627/bdc892547cceb2ef34dedfee80b7006/renderings/color/color1.png'
    # voxel_file = 'results/03001627/bdc892547cceb2ef34dedfee80b7006/raycasted_voxel/voxel.ply'

    # stool
    cam_file = 'results/04379243/142060f848466cad97ef9a13efb5e3f7/renderings/camera/cam3.json'
    image_file = 'results/04379243/142060f848466cad97ef9a13efb5e3f7/renderings/color/color3.png'
    voxel_file = 'results/04379243/142060f848466cad97ef9a13efb5e3f7/raycasted_voxel/voxe3.ply'

    im = Image.open(image_file)
    np_img = np.array(im)
    np_img = np_img[:, :, 0:3]

    # saving the values in the image
    # image_txt_file = 'results/02747177/85d8a1ad55fa646878725384d6baf445/text/color1.txt'
    # with open(image_txt_file, "w") as f:
    #     for u in range(512):
    #         for v in range(512):
    #             if not np.all(np_img[v,u,:] == 255):
    #                 f.write("%d %d %d %d %d\n" % (u, v, np_img[v,u,0], np_img[v,u,1], np_img[v,u,2]))

    cam, cam_pose = load_camera(cam_file)
    voxel_grid = create_voxel_grid(cam)
    raycast(voxel_grid, cam, np_img)

    # Since we have done raycasting in camera system, apply the cam to world transform while saving the voxel
    voxel_grid.save_as_ply(voxel_file, cam_pose)
