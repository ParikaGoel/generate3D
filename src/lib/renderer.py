import os
import click
import trimesh
import pyrender
import numpy as np
from scipy.spatial.transform import Rotation


# Set the intrinsic and extrinsic properties of the camera
def set_custom_camera(scene):
    camera = trimesh.scene.Camera("Cam1", resolution=[512, 512], focal=[525.0, 525.0], z_near=0.01, z_far=1000.0)
    camera_transform = np.array([[1.00000000, 0.00000000, 0.00000000, 0.000457000000],
                                 [0.00000000, 1.00000000, 0.00000000, 0.0115150000],
                                 [0.00000000, 0.00000000, 1.00000000, 1.21847893],
                                 [0.00000000, 0.00000000, 0.00000000, 1.00000000]])
    scene.camera = camera
    scene.camera_transform = camera_transform


# Rotate the scene along a particular axis by specified degrees/ radians
# To apply multiple rotations, provide arguments like below:
# r = R.from_euler('zyx', [[90, 0, 0],[0, 45, 0],[45, 60, 30]], degrees=True)
# Above example stacks three rotations specified degree along all the three rotation axis
def rotate_scene(scene, axis, value, degree=True):
    rot_matrix = Rotation.from_euler(axis, value, degrees=degree)
    transform = np.eye(4)
    # keep the translation same
    transform[0:3, 3] = scene.camera_transform[0:3, 3]
    # update the rotation matrix to the calculated rotation matrix
    transform[0:3, 0:3] = rot_matrix.as_dcm()
    # apply the transformation to the scene
    scene.camera_transform = transform


# Save the scene as png image
def render_scene_as_png(scene, output_folder, _static={'counter': 0}):
    _static['counter'] += 1
    file_name = '%s.png' % _static['counter']
    path = os.path.join(output_folder, file_name)
    png_image = scene.save_image()
    file = open(path, 'wb')
    file.write(png_image)
    file.close()


def generate_data(model_file, output_folder, num=5):
    count = 1
    axis = 'z'
    degree_val = 0
    scene = trimesh.load(model_file, file_type='obj')

    # Define the camera and set the camera of the scene to custom camera
    set_custom_camera(scene)

    while count <= num:
        render_scene_as_png(scene, output_folder)
        degree_val -= 2
        rotate_scene(scene, axis, degree_val)
        count += 1


###############################################################################
# Command line interface.
###############################################################################
@click.command()
@click.argument('obj_file',
                type=click.Path(exists=True, dir_okay=True, readable=True))
@click.option('--out_name',
              type=click.Path(dir_okay=True, writable=True),
              help='The result location to use. By default, use `renderings',
              default='renderings')
@click.option('--num_renderings',
               type=click.INT,
               help='Number of images to render. Default: 5',
               default=5)
def main(obj_file, out_name, num_renderings):
    # obj_file = '/home/parika/WorkingDir/complete3D/data/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/models/model_normalized.obj'
    # renderings_folder = '/home/parika/WorkingDir/complete3D/data/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/renderings'

    generate_data(obj_file, out_name, num_renderings)


if __name__ == '__main__':
    main()

    # from trimesh.viewer.windowed import *
    #
    # model_file = '/home/parika/WorkingDir/complete3D/data/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/models/model_normalized.obj'
    # depth_file = '/home/parika/WorkingDir/complete3D/results/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/renderings/depth/0.png'
    # scene = trimesh.load(model_file, file_type='obj')
    # scene_py = pyrender.Scene.from_trimesh_scene(scene)
    # # viewer = pyrender.Viewer(scene_py, use_raymond_lighting=True)
    #
    # r = pyrender.OffscreenRenderer(512, 512)
    # color, depth = r.render(scene_py)
    # print(color)
    # print(depth)
