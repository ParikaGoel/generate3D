import sys

sys.path.append('../.')
import os
import glob
import time
import torch
import config
import numpy as np
import dataset_loader as loader
import data_formats as formats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def change_color(ply_file, color=(169, 0, 255)):
    vertices, faces = formats.read_ply(ply_file)
    new_vertices = [tuple(vertex[:3]) + color for vertex in vertices]

    formats.write_ply(ply_file, new_vertices, faces)


def grid_to_mesh(grid_lst, ply_file, grid_size=None):
    cube_verts = np.array([[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
                           [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]])  # 8 points

    cube_faces = np.array([[0, 1, 2], [2, 3, 0], [1, 5, 6], [6, 2, 1], [7, 6, 5], [5, 4, 7],
                           [4, 0, 3], [3, 7, 4], [4, 5, 1], [1, 0, 4], [3, 2, 6], [6, 7, 3]])  # 6 faces (12 triangles)

    verts = []
    faces = []
    curr_vertex = 0

    if grid_size is None:
        grid_size = 1

    min_bound = np.array([-grid_size / 2, -grid_size / 2, -grid_size / 2])
    voxel_scale = grid_size / config.vox_dim

    for data in grid_lst:
        grid_coord = np.array((data[0], data[1], data[2])).astype(int)
        grid_color = np.array((data[3], data[4], data[5])).astype(int)
        i = grid_coord[0]
        j = grid_coord[1]
        k = grid_coord[2]

        for cube_vert in cube_verts:
            vertex = (cube_vert * 0.45 + np.array([i, j, k])).astype(float)
            # vertex = (cube_vert + np.array([i, j, k])).astype(float)
            vertex *= voxel_scale
            vertex += min_bound
            vertex = np.append(vertex, grid_color)
            vertex = list(vertex)
            verts.append(vertex)

        for cube_face in cube_faces:
            face = curr_vertex + cube_face
            faces.append(list(face))

        curr_vertex += len(cube_verts)

    formats.write_ply(ply_file, verts, faces)
    return verts, faces


def df_to_mesh(filename, df, trunc=1.0, color=None):
    mask = torch.ge(df, 0.0) & torch.le(df, trunc)

    if not mask.any():
        return

    positions = np.where(mask.cpu().numpy())
    grid_lst = []
    if color is None:
        color = np.array([169, 0, 255])

    for i, j, k in zip(*positions):
        data = np.array((i, j, k, color[0], color[1], color[2]))
        grid_lst.append(data)

    grid_to_mesh(grid_lst, filename)


def occ_to_mesh(filename, occ, color=None):
    mask = torch.eq(occ, 1.0)

    if not mask.any():
        return

    positions = np.where(mask.cpu().numpy())
    grid_lst = []
    if color is None:
        color = np.array([169, 0, 255])

    for i, j, k in zip(*positions):
        data = np.array((i, j, k, color[0], color[1], color[2]))
        grid_lst.append(data)

    grid_to_mesh(grid_lst, filename)


def preprocess_occ(occ, pred=True):
    '''
    preprocess occupancy from network to convert predicted occupancy into a binary occupancy grid
    Also remove the channel dimension and transform it into shape (W x H x D)
    '''
    if pred:
        occ = torch.nn.Sigmoid()(occ)
        occupied_mask = torch.ge(occ, 0.5)
        occ[occupied_mask] = 1.0
        occ[~occupied_mask] = 0.0

    occ = torch.transpose(occ[0], 0, 2)
    return occ


def occ_to_df(occ, trunc, pred=True):
    """
    :param occ: occupancy grid as pytorch tensor of shape (1 x D x H x W)
    :return:
        distance field grid as pytorch tensor of shape (1 x D x H x W)
        all the voxels outside the truncation distance are set to trunc
    """
    # occ = preprocess_occ(occ, pred)

    lin_ind = torch.arange(0, 27, dtype=torch.int16).to(device)
    grid_coords = torch.empty(3, lin_ind.size(0), dtype=torch.int16).to(device)
    grid_coords[0] = lin_ind / 9
    grid_coords[1] = (lin_ind - grid_coords[0] * 9) / 3
    grid_coords[2] = lin_ind % 3
    grid_coords = (grid_coords - 1).float()
    kernel = torch.norm(grid_coords, dim=0)

    # initialize with grid distances
    df = torch.full(size=(34, 34, 34), fill_value=float('inf'), dtype=torch.float32).to(device)
    mask = torch.eq(occ, 1)
    df[1:33, 1:33, 1:33][mask] = 0

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

    indices_x = (grid_coords_vol[0, None, :] + grid_coords[0, :, None]).long()
    indices_y = (grid_coords_vol[1, None, :] + grid_coords[1, :, None]).long()
    indices_z = (grid_coords_vol[2, None, :] + grid_coords[2, :, None]).long()

    new_df = torch.full(size=(34, 34, 34), fill_value=float('inf'), dtype=torch.float32).to(device)
    new_df[1:33, 1:33, 1:33] = torch.reshape(
        torch.min(df[indices_x, indices_y, indices_z] + kernel[:, None], dim=0).values, (32, 32, 32))

    while not torch.allclose(df, new_df, rtol=0, atol=10, equal_nan=True):
        df = new_df
        new_df[1:33, 1:33, 1:33] = torch.reshape(
            torch.min(df[indices_x, indices_y, indices_z] + kernel[:, None], dim=0).values, (32, 32, 32))

    # for i in range(grid_coords_vol.size(1)):
    #     df[tuple(grid_coords_vol[:, i])] = torch.min(df[indices_x[:,i], indices_y[:,i], indices_z[:,i]] + kernel)

    df = df[1:33, 1:33, 1:33]

    mask = torch.gt(df, trunc)
    df[mask] = trunc
    df = torch.transpose(df, 0, 2).unsqueeze(0)

    return df


def occs_to_dfs(occs, trunc, pred=True):
    '''
    generate batch of distance field grids from batch of occupancy grids
    '''
    batch_size = occs.shape[0]
    dfs = torch.stack([occ_to_df(occs[b], trunc, pred) for b in range(batch_size)])
    return dfs


def save_predictions(output_path, names, pred_dfs, target_dfs, pred_occs, target_occs):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for k in range(len(names)):
        name = names[k]
        if pred_dfs is not None:
            pred_df = pred_dfs[k]
            # swaps width and depth dimension and removes the channel dimension
            pred_df = torch.transpose(pred_df[0], 0, 2)
            np.save(os.path.join(output_path, name + "_pred_df"), pred_df.cpu().numpy())
            # save the occupancy mesh from pred distance field
            df_to_mesh(os.path.join(output_path, name + "_pred_mesh.ply"), pred_df, trunc=1.0)

        if target_dfs is not None:
            target_df = target_dfs[k]
            target_df = torch.transpose(target_df[0], 0, 2)
            np.save(os.path.join(output_path, name + "_target_df"), target_df.cpu().numpy())
            df_to_mesh(os.path.join(output_path, name + "_target_mesh.ply"), target_df, trunc=1.0,
                       color=np.array([0, 169, 255]))

        if pred_occs is not None:
            pred_occ = preprocess_occ(pred_occs[k], pred=True)
            occ_to_mesh(os.path.join(output_path, name + "_pred_mesh.ply"), pred_occ)
            np.save(os.path.join(output_path, name + "_pred_mesh"), pred_occ.cpu().numpy())

        if target_occs is not None:
            target_occ = preprocess_occ(target_occs[k], pred=False)
            occ_to_mesh(os.path.join(output_path, name + "_target_mesh.ply"), target_occ)


if __name__ == '__main__':
    colors = {
    "color_Net3_occ" : (169, 255, 0), # Net3; occ
    "color_Net3_tdf" : (255, 255, 0), # Net3; tdf
    "color_Net3_tdf_log" : (255, 0, 255), # Net3; tdf_log
    "color_Net4_occ" : (169, 169, 0),  # Net4; occ
    "color_Net4_tdf" : (255, 0, 169),  # Net4; tdf
    "color_Net4_tdf_log" : (255, 0, 0)  # Net4; tdf_log
    }

    for ply_file in sorted(glob.glob(
            "/home/parika/WorkingDir/complete3D/Assets_remote/output-network/vis/*/*/*/*_pred_mesh.ply")):
        model = ply_file.split("/")[8]
        data = ply_file.split("/")[9]
        color_key = "color_" + model + "_" + data
        change_color(ply_file, colors[color_key])
