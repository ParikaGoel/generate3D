import sys

sys.path.append('../.')
import os
import glob
import json
import torch
import config
import imageio
import JSONHelper
import voxel_grid
import numpy as np
import data_utils as utils


def load_imgs(folder, color = False):
    '''
    :param folder: folder path containing rendered images
    :return:
        returns a tensor of shape [N x (height * width)]
        or [N x (3 * height * width)]
        each row in tensor contains flattened image tensor
    '''
    imgs = []
    for filename in sorted(os.listdir(folder)):
        img = load_img(os.path.join(folder, filename), color)
        imgs.append(img)
    imgs = torch.stack(imgs)
    return imgs


def load_img(png_file, color = False):
    """
    Loads the image from the png file
    If color is False, preprocess it to get the binary mask
    binary mask -> h x w grid containing values (0,1)
    :param png_file: Image png file
    :return:
        image silhouette pytorch tensor (shape: [H * W]) (type: float) if color is False
        color image pytorch tensor (shape: [H * W * 3]) (type: float) if color is True
    """
    img = imageio.imread(png_file)

    if not color:
        img = img[:, :, 0]  # we dont need the color values for binary mask

        # img will contain the occupancy of the pixels now : 1 refers to object in the pixel
        img[img < 255] = 1
        img[img == 255] = 0

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

    color_grid = torch.from_numpy(color_grid / 256)
    return color_grid


def load_occ(txt_file):
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


def load_df(file):
    """
        loads the df grid from the file and returns it as a pytorch tensor
        :param input_file: File storing df values
        :return: df grid as pytorch tensor (shape: [1,D,H,W])
    """
    with open(file, 'rb') as f:
        dims = np.fromfile(f, dtype=np.int32, count=3)
        res = np.fromfile(f, dtype=np.float32, count=1)
        grid2world = np.fromfile(f, dtype=np.float32, count=16)

        n_size = dims[0] * dims[1] * dims[2]
        df = np.fromfile(f, dtype=np.float32, count=n_size)

    df = np.asarray(df).reshape(dims[0], dims[1], dims[2])
    df = torch.from_numpy(df).unsqueeze(0)  # <- adds channel dimension

    # set all distance greater than truncation to truncation distance
    mask = torch.gt(df, config.trunc_dist)
    df[mask] = config.trunc_dist

    return df


def load_occ_df(filename):
    df = np.load(filename)

    df = torch.from_numpy(df)
    mask = torch.gt(df, config.trunc_dist)
    df[mask] = config.trunc_dist
    df = torch.transpose(df, 0, 2).unsqueeze(0)

    return df


def load_sdf(filename):
    sdf = torch.from_numpy(np.load(filename))
    trunc = config.trunc_dist / 32
    mask = torch.lt(sdf, -trunc) & torch.gt(sdf, trunc)
    sdf[mask] = trunc
    sdf = torch.transpose(sdf, 0, 2).unsqueeze(0)

    return sdf


class DatasetLoad(torch.utils.data.Dataset):
    def __init__(self, train_list, n_max_samples=-1, transform=None):
        """
        dataset loader class -> reads the input and gt txt files corresponding to synset_id and model_id
        and gives the occupancy grids for input and target
        :param train_list: list of dict entries. dict entry -> {'synset_id':'02933112', 'model_id':'2f0fd2a5e181b82a4267f85fb94fa2e7'}
        :param n_max_samples: max samples to be used
        """
        self._train_list = train_list
        self.n_samples = len(train_list)
        self.transform = transform
        if n_max_samples != -1:
            self.n_samples = min(self.n_samples, n_max_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        synset_id = self._train_list[index]['synset_id']
        model_id = self._train_list[index]['model_id']

        params = JSONHelper.read("../parameters.json")

        input_occ_file = params["shapenet_raytraced"] + synset_id + "/" + model_id + ".txt"
        gt_df_file = params["shapenet_voxelized"] + synset_id + "/" + model_id + "__0__.df"
        gt_occ_file = params["shapenet_voxelized"] + synset_id + "/" + model_id + "__0__.txt"
        gt_occ_df_file = params["shapenet_voxelized"] + synset_id + "/" + model_id + "_occ_df.npy"
        gt_sdf_file = params["shapenet_voxelized"] + synset_id + "/" + model_id + ".npy"

        occ_grid = load_occ(input_occ_file)
        df_gt = load_df(gt_df_file)
        sdf_gt = load_sdf(gt_sdf_file)
        occ_gt = load_occ(gt_occ_file)
        occ_df_gt = load_occ_df(gt_occ_df_file)

        return {'name': model_id, 'occ_grid': occ_grid, 'occ_gt': occ_gt, 'occ_df_gt': occ_df_gt, 'df_gt':df_gt, 'sdf_gt':sdf_gt}