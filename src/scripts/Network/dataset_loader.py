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
    occ_grid = occ_grid.cpu().numpy().transpose(2, 1, 0)

    positions = np.where(occ_grid >= 0.5)
    with open(txt_file, "w") as f:
        for i, j, k in zip(*positions):
            color = np.array([169, 0, 255])
            data = np.column_stack((i, j, k, color[0], color[1], color[2]))
            np.savetxt(f, data, fmt='%d %d %d %d %d %d', delimiter=' ')


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

    return df


def df_to_mesh(file, trunc_dist=1.0, color=None):
    df_grid = torch.from_numpy(np.load(file))
    mask = torch.gt(df_grid, 0.0) & torch.lt(df_grid, trunc_dist)
    positions = np.where(mask.cpu().numpy())

    grid_lst = []

    if color is None:
        color = np.array([169, 0, 255])

    for i, j, k in zip(*positions):
        data = np.array((i, j, k, color[0], color[1], color[2]))
        grid_lst.append(data)

    ply_file = file[:file.rfind('.')] + '.ply'
    voxel_grid.grid_to_mesh(grid_lst, ply_file)


def save_df(filename, df_grid):
    """
        saves the network output in a ply file for visualization
        :param file: file in which to store the output produced by network
        :param df_grid: network output
        """
    df_grid = torch.transpose(df_grid[0], 0, 2)  # <- removes the channel dimension
    df_grid = df_grid.cpu().numpy()
    np.save(filename, df_grid)


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

        # input_img_file = params["shapenet_renderings"] + synset_id + "/" + model_id + "/color/color00.png"
        gt_df_file = params["shapenet_voxelized"] + synset_id + "/" + model_id + "__0__.df"
        input_occ_file = params["shapenet_raytraced"] + synset_id + "/" + model_id + ".txt"
        gt_occ_file = params["shapenet_voxelized"] + synset_id + "/" + model_id + "__0__.txt"
        # gt_imgs_folder = params["shapenet_renderings"] + synset_id + "/" + model_id + "/color"
        # poses_folder = params["shapenet_renderings"] + synset_id + "/" + model_id + "/pose"

        # input_img = torch.reshape(load_img(input_img_file), (1, config.render_img_height, config.render_img_width)).float()
        df_gt = load_df(gt_df_file)
        occ_grid = load_sample(input_occ_file)
        occ_gt = load_sample(gt_occ_file)
        # imgs_gt = load_imgs(gt_imgs_folder, False)
        # poses = load_poses(poses_folder)

        return {'occ_grid': occ_grid, 'occ_gt': occ_gt, 'df_gt':df_gt}
        # return {'occ_grid': occ_grid, 'occ_gt': occ_gt, 'df_gt': df_gt, 'img': input_img, 'imgs_gt': imgs_gt, 'poses': poses}