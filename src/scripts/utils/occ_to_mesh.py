import os
import numpy as np
from formats import data_formats

def load_occ_txt(file, dims):
    """
        loads the occupancy grid from the text file and returns it as a numpy array
        :param input_file: Text file storing grid coordinates and corresponding color values
        :dims dimensions of the voxel grid
        :return: occupancy grid as numpy array (shape: [W, H, D])
        """
    voxel = np.loadtxt(file, dtype=int)
    occ_grid = np.zeros((dims[0], dims[1], dims[2])).astype(int)

    for data in voxel:
        grid_coord = np.array((data[0], data[1], data[2])).astype(int)
        occ_grid[grid_coord[2], grid_coord[1], grid_coord[0]] = 1

    return occ_grid


def occ_to_mesh(file, out_folder, dims, format='ply'):
    filename = file[file.rfind('/')+1:file.rfind('.txt')+1]
    out_file = os.path.join(out_folder, filename+format)

    occ_grid = load_occ_txt(file, dims)
    positions = np.where(occ_grid == 1)

    with open(txt_file, "w") as f:
        for i, j, k in zip(*positions):
            color = np.array([169, 0, 255])
            data = np.column_stack((i, j, k, color[0], color[1], color[2]))
            np.savetxt(f, data, fmt='%d %d %d %d %d %d', delimiter=' ')


