import os
import glob
import shutil
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
        scene.add(mesh, pose=np.eye(4))
    elif (isinstance(obj_mesh, trimesh.Scene)):
        meshes = obj_mesh.dump()
        for m in meshes:
            mesh = pyrender.Mesh.from_trimesh(m)
            scene.add(mesh, pose=np.eye(4))

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


def export_pose_to_json(cam, pose, file_name):
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

    pose_folder = 'pose'
    pose_folder = os.path.join(out_folder, pose_folder)
    if not os.path.exists(pose_folder):
        os.mkdir(pose_folder)

    scene = create_scene(obj_file)
    mesh_nodes = scene.get_nodes()

    # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera = pyrender.IntrinsicsCamera(fx=focal, fy=focal,
                                       cx=render_img_width / 2,
                                       cy=render_img_height / 2,
                                       znear=znear, zfar=zfar)

    # cam_pose = np.eye(4)
    # cam_pose[:3, 3] = [0.0, 0.0, cam_depth]

    rot_mat = "0.9659258 0.0 -0.2588190 0.0 0.0 1.0 0.0 0.0 0.2588190 0.0 0.9659258 0.0 0.0 0.0 0.0 1.0"
    rot_lst = np.fromstring(rot_mat, dtype=float, sep=' ').reshape(4, 4)
    TranslationVector = "0.0 0.0 1.2 1"
    TranslationVector = np.fromstring(TranslationVector, dtype=float, sep=' ')
    cam_pose = rot_lst
    cam_pose[:, 3] = TranslationVector

    # rot_mat = "0.821092 0.00408988 -0.570781 0 -0.128338 0.975693 -0.177628 0 0.556181 0.219102 0.801659 0 0 0 0 1 "
    # rot_lst = np.fromstring(rot_mat, dtype=float, sep=' ').reshape(4,4)
    # TranslationVector = "-0.954648 -0.366605 1.38289 1"
    # TranslationVector = np.fromstring(TranslationVector, dtype=float, sep=' ')
    # cam_pose = rot_lst
    # cam_pose[:,3] = TranslationVector

    # pose -> gives the pose of the camera in the world system; camera to world transformation
    scene.add(camera, pose=cam_pose)

    pyrender.Viewer(scene)

    count = 0
    axis = 'y'
    deg = 0
    trans = 0
    while count < num_renderings:
        color_file = os.path.join(color_img_folder, 'color%s.png' % count)
        depth_file = os.path.join(depth_img_folder, 'depth%s.png' % count)
        pose_file = os.path.join(pose_folder, 'pose%s.json' % count)

        rot_matrix = Rotation.from_euler(axis, deg, degrees=True)
        pose = np.eye(4)
        pose[0:3, 0:3] = rot_matrix.as_dcm()
        pose[:3, 3] = [0.0, 0.0, 0.0]

        for mesh in mesh_nodes:
            mesh.matrix = pose

        # pyrender.Viewer(scene, show_world_axis=True)
        export_pose_to_json(camera, pose, pose_file)
        #
        r = pyrender.OffscreenRenderer(render_img_width, render_img_height)
        color, depth = r.render(scene)
        # # mask = depth != 0
        # # color = mask[:, :, None] * color
        save_img(color, depth, color_file, depth_file)
        # show(color, depth)

        deg += 15
        trans += 0.05
        count += 1


# if __name__ == '__main__':
#     num_renderings = 10
#
#     params = JSONHelper.read("./parameters.json")
#
#     failed_cases = {}
#     failed_ids = []
#     file = "./failed_cases.json"
#
#     for f in glob.glob(params["shapenet"] + "/**/*/models/model_normalized.obj"):
#         catid_cad = f.split("/", 6)[4]
#         id_cad = f.split("/", 6)[5]
#
#         if not catid_cad in failed_cases.keys():
#             failed_cases[catid_cad] = []
#
#         print("catid: ", catid_cad, " , id: ", id_cad)
#
#         try:
#             outdir = params["shapenet_renderings"] + "/" + catid_cad + "/" + id_cad
#             pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
#
#             generate_images(f, outdir, num_renderings)
#
#             dir = params["shapenet"] + "/" + catid_cad + "/" + id_cad
#             shutil.rmtree(dir)
#         except:
#             failed_cases[catid_cad].append(id_cad)
#             pass
#
#         JSONHelper.write(file, failed_cases)

if __name__ == '__main__':
    num_renderings = 24

    params = JSONHelper.read("./parameters.json")

    failed_cases = {}
    failed_ids = []
    file = "/media/sda2/shapenet/renderings/failed_cases.json"

    synset_id = "02747177"
    model_id = "fd013bea1e1ffb27c31c70b1ddc95e3f"

    obj_file = params["shapenet"] + "/" + synset_id + "/" + model_id + "/models/model_normalized.obj"
    # outdir = params["shapenet_renderings"] + "/" + synset_id + "/" + model_id
    outdir = "/media/sda2/shapenet/renderings" + "/" + synset_id + "/" + model_id
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    generate_images(obj_file, outdir, num_renderings)

    # for f in glob.glob(params["shapenet"] + "/**/*/models/model_normalized.obj"):
    # # for f in glob.glob(params["shapenet"] + synset_id + "/*/models/model_normalized.obj"):
    #     synset_id = f.split("/", 8)[5]
    #     model_id = f.split("/", 8)[6]
    #     print("synset_id: ", synset_id, " , model_id: ", model_id)
    #     try :
    #         obj_file = params["shapenet"] + "/" + synset_id + "/" + model_id + "/models/model_normalized.obj"
    #         outdir = "/media/sda2/shapenet/renderings" + "/" + synset_id + "/" + model_id
    #
    #         if os.path.exists(outdir):
    #             print(synset_id, " : ", model_id, " already rendered. Skipping......")
    #             continue
    #
    #         pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    #
    #         generate_images(obj_file, outdir, num_renderings)
    #
    #     except:
    #         print("in exception block\n")
    #         failed_ids.append(model_id)
    #         pass
    #
    # failed_cases[model_id] = failed_ids
    # JSONHelper.write(file, failed_cases)

