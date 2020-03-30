import sys
sys.path.append('./Network')
import glob
import torch
import config
import JSONHelper
import numpy as np
import dataset_loader as loader

params = JSONHelper.read("./parameters.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def occ_to_df(occ, trunc):
    lin_ind = torch.arange(0, 27, dtype=torch.int16).to(device)
    grid_coords = torch.empty(3, lin_ind.size(0), dtype=torch.int16).to(device)
    grid_coords[0] = lin_ind / 9
    grid_coords[1] = (lin_ind - grid_coords[0] * 9) / 3
    grid_coords[2] = lin_ind % 3
    grid_coords = (grid_coords - 1).float()
    kernel = torch.norm(grid_coords, dim=0)

    # initialize with grid distances
    init_df = torch.full(size=occ.shape, fill_value=float('inf'), dtype=torch.float32).to(device)
    mask = torch.eq(occ, 1)
    init_df[mask] = 0
    df = torch.full(size=(34,34,34), fill_value=float('inf'), dtype=torch.float32).to(device)
    df[1:33,1:33,1:33] = init_df

    lin_ind_volume = torch.arange(0, 34 * 34 * 34,
                                  out=torch.LongTensor()).to(device)
    grid_coords_vol = torch.empty(3, lin_ind_volume.size(0)).to(device)
    grid_coords_vol[0] = lin_ind_volume / (34 * 34)
    tmp = lin_ind_volume - (grid_coords_vol[0] * 34 * 34).long()
    grid_coords_vol[1] = tmp / 34
    grid_coords_vol[2] = torch.remainder(tmp, 34)

    cal_mask = torch.gt(grid_coords_vol, 0) & torch.lt(grid_coords_vol, 33)
    cal_mask = cal_mask[0] & cal_mask[1] & cal_mask[2]
    grid_coords_vol = grid_coords_vol[:, cal_mask].long()

    indices_x = (grid_coords_vol[0,None,:] + grid_coords[0, :, None]).long()
    indices_y = (grid_coords_vol[1,None,:] + grid_coords[1, :, None]).long()
    indices_z = (grid_coords_vol[2,None,:] + grid_coords[2, :, None]).long()

    for i in range(grid_coords_vol.size(1)):
        df[tuple(grid_coords_vol[:, i])] = torch.min(df[indices_x[:,i], indices_y[:,i], indices_z[:,i]] + kernel)

    df = df[1:33, 1:33, 1:33]

    mask = torch.gt(df, trunc)
    df[mask] = trunc+1

    return df


if __name__ == '__main__':

    # for f in glob.glob(params["shapenet_voxelized"] + config.synset_id + "/*.txt"):
    #     print(f)
    #     occ_grid = loader.load_sample(f)
    #     occ_grid = torch.transpose(occ_grid[0], 1, 2)
    #     df = occ_to_df(occ_grid, config.trunc_dist)
    #     df_file = f[:f.rfind("__0__")] + "_occ_df"
    #     np.save(df_file, df.cpu().numpy())

    occ_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-voxelized-gt/03001627/1a6f615e8b1b5ae4dbbc9440457e303e__0__.txt"
    df_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-voxelized-gt/03001627/1a6f615e8b1b5ae4dbbc9440457e303e_occ_2_df.ply"
    occ_grid = loader.load_occ(occ_file)
    df = occ_to_df(occ_grid, 4.0, False)
    df_to_mesh(df_file, df, 1.0)