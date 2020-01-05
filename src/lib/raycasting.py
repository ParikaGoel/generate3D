import json
import trimesh
import numpy as np
from camera import *
from PIL import Image
from voxel_grid import *


# Debugging helper function : saves the color values with corresponding pixel coordinates
# Considers image to have white background color : so we dont save those values
def save_image_info_txt(image, file):
    with open(file, "w") as f:
        for u in range(image.shape[0]):
            for v in range(image.shape[1]):
                if not np.all(image[v,u,:] == 255):
                    f.write("%d %d %d %d %d\n" % (u, v, image[v,u,0], image[v,u,1], image[v,u,2]))


# def load_camera(cam_file):
#     with open(cam_file) as camera_file:
#         data = json.load(camera_file)
#
#         # read the intrinsic parameters
#         resolution = [data['intrinsic']['width'], data['intrinsic']['height']]
#         focal = [data['intrinsic']['fx'], data['intrinsic']['fy']]
#         znear = data['intrinsic']['z_near']
#         zfar = data['intrinsic']['z_far']
#
#         cam = trimesh.scene.Camera(resolution=resolution, focal=focal, z_near=znear, z_far=zfar)
#
#         # read the camera pose
#         pose = np.empty(0)
#         for val in data['pose']:
#             pose = np.append(pose, val)
#         pose = np.transpose(np.reshape(pose, (4, 4)))
#
#         return cam, pose


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
    voxel_txt_file = 'results/04379243/142060f848466cad97ef9a13efb5e3f7/text/voxel.txt'
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
    cam_file = 'results/04379243/142060f848466cad97ef9a13efb5e3f7/renderings/camera/cam1.json'
    image_file = 'results/04379243/142060f848466cad97ef9a13efb5e3f7/renderings/color/color1.png'
    voxel_file = 'results/04379243/142060f848466cad97ef9a13efb5e3f7/raycasted_voxel/voxel1.ply'
    voxel_txt_file = 'results/04379243/142060f848466cad97ef9a13efb5e3f7/text/voxelinfo.txt'

    im = Image.open(image_file)
    np_img = np.array(im)
    np_img = np_img[:, :, 0:3]

    cam = load_camera(cam_file)
    voxel_grid = create_voxel_grid(cam)
    raycast(voxel_grid, cam, np_img)

    voxel_grid.save_vox(voxel_txt_file)

    # Since we have done raycasting in camera system, apply the cam to world transform while saving the voxel
    voxel_grid.to_mesh(voxel_file, cam.pose)
