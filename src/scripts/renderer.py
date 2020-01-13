import os
import glob
import pathlib
import trimesh
import pyrender
import numpy as np
from camera import *
from config import *
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import JSONHelper


def create_scene(obj_file):
    obj_mesh = trimesh.load(obj_file)
    scene = pyrender.Scene(ambient_light=[1., 1., 1.], bg_color=color_bg)  # bg = {0, 145, 255}

    if (isinstance(obj_mesh, trimesh.Trimesh)):
        mesh = pyrender.Mesh.from_trimesh(obj_mesh)
        scene.add(mesh)
    elif (isinstance(obj_mesh, trimesh.Scene)):
        meshes = obj_mesh.dump()
        for m in meshes:
            mesh = pyrender.Mesh.from_trimesh(m)
            scene.add(mesh)

    return scene


# ToDo: improvement -> try saving in tif (TIFF library)
# tiff.imsave
def save_img(color, depth, color_img_file, depth_img_file):
    depth = (depth * depth_factor)
    depth = depth.astype(np.uint8)
    im = Image.fromarray(color)
    depth_im = Image.fromarray(depth, mode='L')
    im.save(color_img_file)
    depth_im.save(depth_img_file)


def export_cam_to_json(cam, pose, file_name):
    pose = pose.flatten(order='F')
    cam_data = \
        {
            'intrinsic':
                {
                    'cx': float(cam.cx),
                    'cy': float(cam.cy),
                    'fx': float(cam.fx),
                    'fy': float(cam.fy),
                    'z_near': float(cam.znear),
                    'z_far': float(cam.zfar)
                },
            'pose': pose.tolist()
        }

    with open(file_name, 'w') as fp:
        json.dump(cam_data, fp)


def show(color_img, depth_img):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(color_img)
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(depth_img, cmap=plt.cm.gray_r)
    plt.show()
    plt.close(fig)


def generate_images(obj_file, out_folder, num_renderings):
    # create folders to save renderings
    color_img_folder = 'color'
    color_img_folder = os.path.join(out_folder, color_img_folder)
    if not os.path.exists(color_img_folder):
        os.mkdir(color_img_folder)

    depth_img_folder = 'depth'
    depth_img_folder = os.path.join(out_folder, depth_img_folder)
    if not os.path.exists(depth_img_folder):
        os.mkdir(depth_img_folder)

    cam_folder = 'cam'
    cam_folder = os.path.join(out_folder, cam_folder)
    if not os.path.exists(cam_folder):
        os.mkdir(cam_folder)

    scene = create_scene(obj_file)

    camera = pyrender.IntrinsicsCamera(fx=focal, fy=focal,
                                       cx=render_img_width / 2,
                                       cy=render_img_height / 2,
                                       znear=znear, zfar=zfar)

    count = 0
    axis = 'z'
    deg = 0
    trans = 0
    while count < num_renderings:
        color_file = os.path.join(color_img_folder, 'color%s.png' % count)
        depth_file = os.path.join(depth_img_folder, 'depth%s.png' % count)
        cam_file = os.path.join(cam_folder, 'cam%s.json' % count)

        rot_matrix = Rotation.from_euler(axis, deg, degrees=True)
        pose = np.eye(4)
        pose[0:3, 0:3] = rot_matrix.as_dcm()
        pose[:3, 3] = [0.0, 0.0, cam_depth]
        cam_node = scene.add(camera, pose=pose)

        # pyrender.Viewer(scene)
        export_cam_to_json(camera, pose, cam_file)

        r = pyrender.OffscreenRenderer(render_img_width, render_img_height)
        color, depth = r.render(scene)
        # mask = depth != 0
        # color = mask[:, :, None] * color
        save_img(color, depth, color_file, depth_file)
        # show(color, depth)

        scene.remove_node(cam_node)

        deg -= 1
        trans += 0.05
        count += 1


if __name__ == '__main__':
    num_renderings = 10

    params = JSONHelper.read("./parameters.json")

    for f in glob.glob(params["shapenet"] + "/**/*/models/model_normalized.obj"):
        catid_cad = f.split("/", 6)[4]
        id_cad = f.split("/", 6)[5]

        outdir = params["shapenet_renderings"] + "/" + catid_cad + "/" + id_cad
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

        generate_images(f, outdir, num_renderings)
