import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from model import *
from camera import *
from voxel_grid import *


# function to show an image
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()


def main():
    # stool
    cam_file = 'results/04379243/142060f848466cad97ef9a13efb5e3f7/renderings/camera/cam1.json'
    image_file = 'results/04379243/142060f848466cad97ef9a13efb5e3f7/renderings/color/color1.png'
    voxel_file = 'results/04379243/142060f848466cad97ef9a13efb5e3f7/raycasted_voxel/voxel_model.ply'
    voxel_txt_file = 'results/04379243/142060f848466cad97ef9a13efb5e3f7/text/voxelinfo.txt'

    # im = Image.open(image_file)
    # np_img = np.array(im)
    # np_img = np_img[:, :, 0:3]

    cam = load_camera(cam_file)
    voxel_grid = create_voxel_grid(cam)
    voxel_grid.load_vox(voxel_txt_file)

    input = voxel_grid.color_grid
    input = np.transpose(input)

    input_tensor = torch.from_numpy(input)
    input_tensor = input_tensor.unsqueeze(0)
    net = Net(3, 3)
    net = net.float()
    out = net(input_tensor.float())
    out_np = out.detach().numpy()
    out_np = out_np[0, :, :, :, :]
    out_np = np.transpose(out_np)

    voxel_grid.occ_grid = np.zeros((32,32,32),dtype=int)
    voxel_grid.color_grid = np.zeros((32,32,32,3),dtype=int)

    color = np.array((0,0,255))

    for i in range(32):
        for j in range(32):
            for k in range(32):
                if not np.all(out_np[i,j,k,:] == 0):
                    print(i, j, k, out_np[i,j,k,:])
                    grid_coord = np.array((i,j,k))
                    voxel_grid.set_color(grid_coord, color)

    voxel_grid.to_mesh(voxel_file)




if __name__ == '__main__':
    main()