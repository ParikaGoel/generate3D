import os
import math
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision.transforms as transforms

import JSONHelper
from model import *
from camera import *
from config import *
from voxel_grid import *


# function to show an image
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def txt_to_input_occ_grid(input_file):
    voxel = np.loadtxt(input_file, dtype=int)
    occ_grid = np.zeros((vox_dim, vox_dim, vox_dim)).astype(int)

    for data in voxel:
        grid_coord = np.array((data[0], data[1], data[2])).astype(int)
        occ_grid[grid_coord[0], grid_coord[1], grid_coord[2]] = 1

    mean = np.mean(occ_grid)
    std = np.std(occ_grid)

    occ_grid = torch.from_numpy(occ_grid.transpose(2, 1, 0))
    occ_grid = occ_grid.float()
    if std != 0:
        occ_grid.sub(mean).div(std)
    else:
        occ_grid.sub(mean)
    return occ_grid, mean, std


def output_occ_grid_to_txt(output_file, occ_grid, mean, std):
    occ_grid = occ_grid[0]
    if std != 0:
        occ_grid.mul(std).add(mean)
    else:
        occ_grid.add(mean)

    occ_grid = occ_grid.clamp(0, 1)
    occ_grid = occ_grid.numpy().transpose(2, 1, 0)

    positions = np.where(occ_grid > 0.5)
    with open(output_file, "w") as f:
        for i, j, k in zip(*positions):
            color = np.array([169, 0, 255])
            data = np.column_stack((i, j, k, color[0], color[1], color[2]))
            np.savetxt(f, data, fmt='%d %d %d %d %d %d', delimiter=' ')




# process the input voxel grid which is a numpy array
# Converts numpy.ndarray (W x H X D X C) in the range [0, 255] to a
# torch.FloatTensor of shape (C x D X H x W) in the range [0.0, 1.0].
# Normalize an tensor image with mean and standard deviation.
#     input[channel] = (input[channel] - mean[channel]) / std[channel]
def colorgrid_to_modelinput(voxel, mean, std):
    voxel = torch.from_numpy(voxel.transpose(3, 2, 1, 0))
    voxel = voxel.float().div(255)
    for t, m, s in zip(voxel, mean, std):
        if s != 0:
            t.sub_(m).div_(s)
    return voxel


# process the input voxel grid which is a numpy array
# Converts numpy.ndarray (W x H X D) with values [0, 1] to a
# torch.FloatTensor of shape (D X H x W) in the range [0.0, 1.0].
# Normalize an tensor image with mean and standard deviation.
#     input = (input - mean) / std
def occgrid_to_modelinput(voxel, mean, std):
    voxel = torch.from_numpy(voxel.transpose(2, 1, 0))
    voxel = voxel.float()
    if std != 0:
        voxel.sub(mean).div(std)
    else:
        voxel.sub(mean)
    return voxel


def get_model_input(voxel_grid, voxel_file):
    voxel_grid.load_vox(voxel_file)

    mean = np.mean(voxel_grid.color_grid, axis=(0, 1, 2)) / 255
    std = np.std(voxel_grid.color_grid, axis=(0, 1, 2)) / 255
    input = colorgrid_to_modelinput(voxel_grid.color_grid, mean, std)

    return input, mean, std

def get_model_input_occ(voxel_grid, voxel_file):
    voxel_grid.load_vox(voxel_file)

    mean = np.mean(voxel_grid.occ_grid)
    std = np.std(voxel_grid.occ_grid)
    input = occgrid_to_modelinput(voxel_grid.occ_grid, mean, std)

    # positions = np.where(input == 1.0)
    # for i,j,k in zip(*positions):
    #     print(i,j,k)

    return input, mean, std


# process the output from the network which is a torch.FloatTensor of shape (C x D X H x W)
# First denormalize the tensor image with mean and standard deviation
#       output[channel] = (output[channel] * std[channel]) + mean[channel]
# This will give tensor values in the range [0.0, 1.0]
# Then convert it into numpy.ndarray (W x H X D X C) in the range [0, 255]
def modeloutput_to_colorgrid(voxel, mean, std):
    for t, m, s in zip(voxel, mean, std):
        t.mul_(s).add_(m)

    voxel = voxel.clamp(0, 1)
    voxel = voxel.mul(255).int()
    voxel = voxel.numpy().transpose(3, 2, 1, 0)
    return voxel

# process the output from the network which is a torch.FloatTensor of shape (1 x D X H x W)
# First denormalize the tensor image with mean and standard deviation
#       output = (output * std) + mean
# This will give tensor values in the range [0.0, 1.0]
# Then convert it into numpy.ndarray (W x H X D) with values [0,1]
def modeloutput_to_occgrid(voxel, mean, std):
    voxel = voxel[0]
    if std != 0:
        voxel.mul(std).add(mean)
    else:
        voxel.add(mean)

    voxel = voxel.clamp(0, 1)
    voxel = voxel.numpy().transpose(2, 1, 0)
    return voxel


# takes the output of the model and converts it into a voxelized mesh
def process_output(output_folder, model_output, mean, std):
    batch_size = model_output.size()[0]
    if model_output.size()[1] == 1:
        occ = True
    else:
        occ = False

    for nVoxel in range(batch_size):
        voxel = model_output[nVoxel]
        if occ:
            output_occ_grid_to_txt(output_folder+"output.txt",voxel, mean, std)
        else:
            voxel = modeloutput_to_colorgrid(voxel, mean, std)


def train(input, target):
    model = Net(1, 1).cpu()
    distance = nn.MSELoss()
    # distance = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

    input = Variable(input).cpu()
    target = Variable(target).cpu()

    for epoch in range(num_epochs):
        # ===================forward=====================
        output = model(input)
        loss = distance(output, target)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================log========================
        if (epoch + 1) % 10 == 0:
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    return output


def main():
    # stool
    catid = "04379243"
    id = "142060f848466cad97ef9a13efb5e3f7"

    params = JSONHelper.read("./parameters.json")

    cam_file = params["shapenet_renderings"] + catid + "/" + id + "/cam/cam0.json"
    input_file = params["shapenet_raytraced"] + catid + "/" + id + ".txt"
    target_file = params["shapenet_voxelized"] + catid + "/" + id + "__0__.txt"
    output_folder = params["network_output"]

    input, mean, std = txt_to_input_occ_grid(input_file)
    input = input.unsqueeze(0).unsqueeze(0)
    target, mean_target, std_target = txt_to_input_occ_grid(target_file)
    target = target.unsqueeze(0)

    output = train(input, target)
    output = output.detach()
    process_output(output_folder, output, mean, std)
    txt_to_mesh(output_folder+"output.txt",output_folder+"output.ply")

    # cam = load_camera(cam_file)
    # voxel_grid = create_voxel_grid(cam)
    # voxel_grid.load_vox(target_file)
    # voxel_grid.to_mesh(target_mesh, cam.pose)
    # input, mean, std = get_model_input(voxel_grid, voxel_txt_file)
    # input, mean, std = get_model_input_occ(voxel_grid, voxel_txt_file)
    # input = input.unsqueeze(0).unsqueeze(0)
    # # target, mean_target, std_target = get_model_input(voxel_grid, target_file)
    # target, mean_target, std_target = get_model_input_occ(voxel_grid, target_file)
    # target = target.unsqueeze(0)
    #
    # output = train(input, target)
    #
    # output = output.detach()
    # process_output(output_folder, output, mean, std, voxel_grid, cam)


if __name__ == '__main__':
    main()

# classification task