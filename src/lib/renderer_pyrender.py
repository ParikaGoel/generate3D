import numpy as np
import pyrender
import trimesh
from PIL import Image
import matplotlib.pyplot as plt

if __name__=='__main__':

    # generate mesh
    fuze_file = '/home/parika/WorkingDir/complete3D/data/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/models/model_normalized.obj'
    fuze_mesh = trimesh.load(fuze_file)
    mesh = pyrender.Mesh.from_trimesh(list(fuze_mesh.geometry.values()))

    camera = pyrender.IntrinsicsCamera(fx=525.0,fy=525.0,cx = 256, cy = 256, znear=0.5, zfar=1.5)
    light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=2e3)

    pose = [[1.0, 0.0, 0.0, 0.000457000000],
            [0.0, 1.0, 0.0, 0.0115150000],
            [0.0, 0.0, 1.0, 1.21847893],
            [0.0, 0.0, 0.0, 1.0]]

    # compose scene
    scene = pyrender.Scene()

    scene.add(mesh)
    scene.add(camera, pose=pose)

    pyrender.Viewer(scene, use_raymond_lightning=True)

    # render scene
    r = pyrender.OffscreenRenderer(512, 512)
    color, depth = r.render(scene)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(color)
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(depth, cmap=plt.cm.gray_r)
    plt.show()