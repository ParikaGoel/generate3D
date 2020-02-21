import sys

sys.path.append('../.')
import os
import glob
import json
import torch
import config
import imageio
import JSONHelper
import numpy as np


def load_imgs(folder):
    '''
    :param folder: folder path containing rendered images
    :return:
        returns a tensor of shape [N x (height * width)]
        each row in tensor contains flattened image tensor
    '''
    imgs = []
    for filename in sorted(os.listdir(folder)):
        img = load_img(os.path.join(folder, filename))
        imgs.append(img)
    imgs = torch.stack(imgs)
    return imgs

    
def load_img(png_file):
    """
    Loads the image from the png file and preprocess it to get the image silhouette
    image silhouette -> h x w grid containing values (0,1)
    :param png_file: Image png file
    :return:
        image silhouette as pytorch tensor (shape: [H, W]) (type: float)
    """
    img = imageio.imread(png_file)
    img = img[:, :, 0]  # we dont need the color values for silhouette
    img[img < 255] = 1
    img[img == 255] = 0
    # img will contain the occupancy of the pixels now : 1 refers to object in the pixel
    img = torch.flatten(torch.from_numpy(img))
    return img


def load_poses(poses_folder):
    '''
        :param folder: folder path containing rendered poses
        :return:
            returns a tensor of shape [N x 16]
            each row is for flattened pose tensor
    '''
    poses = []
    for filename in sorted(os.listdir(poses_folder)):
        pose = load_pose(os.path.join(poses_folder, filename))
        poses.append(pose)
    poses = torch.stack(poses)
    return poses


def load_pose(pose_file):
    data = JSONHelper.read(pose_file)
    # read the pose
    pose = np.empty(0)
    for val in data['pose']:
        pose = np.append(pose, val)

    pose = np.transpose(np.reshape(pose, (4, 4)))
    pose = torch.from_numpy(pose).float()
    pose[2, 3] = -config.cam_depth
    pose[1, 3] = 0.1
    return pose


def load_color_sample(txt_file):
    """
        loads the color grid from the text file and returns it as a pytorch tensor
        :param input_file: Text file storing grid coordinates and corresponding color values
        :return: occupancy grid as pytorch tensor (shape: [C,D,H,W])
        """
    voxel = np.loadtxt(txt_file, dtype=int)
    color_grid = np.full((3, config.vox_dim, config.vox_dim, config.vox_dim), 256).astype(int)

    for data in voxel:
        grid_coord = np.array((data[0], data[1], data[2])).astype(int)
        color = np.array((data[3], data[4], data[5])).astype(int)
        color_grid[:, grid_coord[2], grid_coord[1], grid_coord[0]] = color

    # color_grid = torch.from_numpy(color_grid.transpose(3, 2, 1, 0))
    color_grid = torch.from_numpy(color_grid)
    return color_grid


def load_sample(txt_file):
    """
    loads the occupancy grid from the text file and returns it as a pytorch tensor
    :param input_file: Text file storing grid coordinates and corresponding color values
    :return: occupancy grid as pytorch tensor (shape: [1,D,H,W])
    """
    voxel = np.loadtxt(txt_file, dtype=int)
    occ_grid = np.zeros((config.vox_dim, config.vox_dim, config.vox_dim)).astype(int)

    for data in voxel:
        grid_coord = np.array((data[0], data[1], data[2])).astype(int)
        occ_grid[grid_coord[2], grid_coord[1], grid_coord[0]] = 1

    occ_grid = torch.from_numpy(occ_grid)
    occ_grid = occ_grid.float().unsqueeze(0)  # <- Adds the channel dimension
    return occ_grid


def load_all_samples(files):
    """
    given a list of input text files, loads occupancy grid for each file
    and returns a list of all the occupancy grids
    :param files: list of text files
    :return: list of occupancy grid as pytorch tensor (shape: [1,D,H,W])
    """
    data = []
    for file in files:
        data.append(load_sample(file))

    return data


def save_sample(txt_file, occ_grid):
    """
    saves the network output in a text file
    :param txt_file: output text file in which to store the output produced by network
    :param occ_grid: network output
    """
    occ_grid = occ_grid[0]  # <- removes the channel dimension
    occ_grid = occ_grid.clamp(0, 1)
    occ_grid = occ_grid.cpu().numpy().transpose(2, 1, 0)

    positions = np.where(occ_grid >= 0.5)
    with open(txt_file, "w") as f:
        for i, j, k in zip(*positions):
            color = np.array([169, 0, 255])
            data = np.column_stack((i, j, k, color[0], color[1], color[2]))
            np.savetxt(f, data, fmt='%d %d %d %d %d %d', delimiter=' ')


class DatasetLoad(torch.utils.data.Dataset):
    def __init__(self, train_list, n_max_samples=-1):
        """
        dataset loader class -> reads the input and gt txt files corresponding to synset_id and model_id
        and gives the occupancy grids for input and target
        :param train_list: list of dict entries. dict entry -> {'synset_id':'02933112', 'model_id':'2f0fd2a5e181b82a4267f85fb94fa2e7'}
        :param n_max_samples: max samples to be used
        """
        self._train_list = train_list
        self.n_samples = len(train_list)
        if n_max_samples != -1:
            self.n_samples = min(self.n_samples, n_max_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        synset_id = self._train_list[index]['synset_id']
        model_id = self._train_list[index]['model_id']

        params = JSONHelper.read("../parameters.json")

        input_occ_file = params["shapenet_raytraced"] + synset_id + "/" + model_id + ".txt"
        gt_occ_file = params["shapenet_voxelized"] + synset_id + "/" + model_id + "__0__.txt"
        gt_imgs_folder = params["shapenet_renderings"] + synset_id + "/" + model_id + "/color"
        poses_folder = params["shapenet_renderings"] + synset_id + "/" + model_id + "/pose"

        occ_grid = load_sample(input_occ_file)
        occ_gt = load_sample(gt_occ_file)
        imgs_gt = load_imgs(gt_imgs_folder)
        poses = load_poses(poses_folder)

        return {'occ_grid': occ_grid, 'occ_gt': occ_gt, 'imgs_gt': imgs_gt, 'poses': poses}


if __name__ == '__main__':
    txt_file = "./../../../Assets/shapenet-raytraced/04379243/142060f848466cad97ef9a13efb5e3f7.txt"
    occ_grid = load_sample(txt_file)
    print(occ_grid.shape)
