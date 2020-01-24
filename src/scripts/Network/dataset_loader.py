import sys
sys.path.append('../.')
import torch
import config
import JSONHelper
import numpy as np


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
        occ_grid[grid_coord[0], grid_coord[1], grid_coord[2]] = 1

    occ_grid = torch.from_numpy(occ_grid.transpose(2, 1, 0))
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

    positions = np.where(occ_grid > 0.5)
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

        input_file = params["shapenet_raytraced"] + synset_id + "/" + model_id + ".txt"
        target_file = params["shapenet_voxelized"] + synset_id + "/" + model_id + "__0__.txt"

        occ_grid = load_sample(input_file)
        gt = load_sample(target_file)

        return {'occ_grid': occ_grid, 'occ_gt': gt}


if __name__ == '__main__':
    txt_file = "./../../../Assets/shapenet-raytraced/04379243/142060f848466cad97ef9a13efb5e3f7.txt"
    occ_grid = load_sample(txt_file)
    print(occ_grid.shape)
