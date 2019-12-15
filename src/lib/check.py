import os
import json
import click
import trimesh
import pyrender
import numpy as np
from scipy.spatial.transform import Rotation


# Set the intrinsic and extrinsic properties of the camera
def set_custom_camera(scene):
    camera = trimesh.scene.Camera("Cam1", resolution=[512, 512], focal=[525.0, 525.0], z_near=0.5, z_far=1.5)
    camera_transform = np.array([[1.00000000, 0.00000000, 0.00000000, 0.000457000000],
                                 [0.00000000, 1.00000000, 0.00000000, 0.0115150000],
                                 [0.00000000, 0.00000000, 1.00000000, 1.21847893],
                                 [0.00000000, 0.00000000, 0.00000000, 1.00000000]])
    scene.camera = camera
    scene.camera_transform = camera_transform


def main():
    model_file = 'data/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/models/model_normalized.obj'
    voxel_file = 'results/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/raycasted_voxel/voxel1.ply'
    scene = trimesh.load(model_file, file_type='obj')
    # Define the camera and set the camera of the scene to custom camera
    set_custom_camera(scene)
    scene.show()

    # voxel = trimesh.load(voxel_file, file_type='ply')
    # print(voxel)

    # origins, vectors, pixels = scene.camera_rays()
    # print("Bounds of scene: ", scene.bounds)
    # print("Bounding Box Size: ", scene.extents)
    # print("Center of bounding box: ", scene.centroid)
    #
    # print("Camera origin: ", origins)
    # print("Ray vectors: ", vectors)
    # print("Pixels: ", pixels)


if __name__ == '__main__':
    main()
