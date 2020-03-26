import sys
sys.path.append('../.')
import os
import torch
import config
import numpy as np
import dataset_loader as loader
from data_formats import write_ply

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    write_ply(ply_file, verts, faces)
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
    mask = torch.eq(torch.from_numpy(occ), 1.0)

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
    '''
    if pred:
        occ = torch.nn.Sigmoid()(occ)
        occupied_mask = torch.ge(occ, 0.5)
        occ[occupied_mask] = 1.0
        occ[~occupied_mask] = 0.0

    occ = torch.transpose(occ[0], 0, 2).numpy()
    return occ


def postprocess_df(df, trunc):
    df = torch.transpose(torch.from_numpy(df), 0, 2).unsqueeze(0)
    mask = torch.gt(df, trunc)
    df[mask] = trunc
    return df


def occ_to_df(occ, trunc, pred=True):
    '''
    :param occ: occupancy grid as numpy array of shape (W x H x D)
    :return:
        distance field grid as numpy array of shape (W x H x D)
        all the voxels outside the truncation distance are set to trunc + 1
    '''
    occ = preprocess_occ(occ, pred)
    width, height, depth = occ.shape

    kernel = np.ndarray(shape=(3, 3, 3), dtype=np.float32)
    for k in range(-1, 2):
        for j in range(-1, 2):
            for i in range(-1, 2):
                kernel[k+1,j+1,i+1] = np.linalg.norm(np.array([k, j, i]))

    # initialize with grid distances
    df = np.full(shape=(occ.shape), fill_value=float('inf'), dtype=np.float32)
    occ_set_inds = occ == 1
    df[occ_set_inds] = 0

    found = True
    while found:
        found = False
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    dmin = df[x,y,z]
                    for k in range(-1, 2):
                        for j in range(-1, 2):
                            for i in range(-1, 2):
                                n = np.array([x + i, y + j, z + k])
                                if n[0] < width and n[1] < height and n[2] < depth:
                                    dcurr = df[n[0],n[1], n[2]] + kernel[i + 1][j + 1][k + 1]
                                    if dcurr < dmin and dcurr <= trunc:
                                        dmin = dcurr
                                        found = True
                    df[x, y, z] = dmin

    df_inds = df > trunc
    df[df_inds] = trunc + 1
    df = postprocess_df(df, trunc)

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
            np.save(os.path.join(output_path,name+"_pred_df"), pred_df.cpu().numpy())
            # save the occupancy mesh from pred distance field
            df_to_mesh(os.path.join(output_path,name+"_pred_mesh.ply"), pred_df, trunc=1.0)
    
        if target_dfs is not None:
            target_df = target_dfs[k]
            target_df = torch.transpose(target_df[0], 0, 2)
            np.save(os.path.join(output_path,name+"_target_df"), target_df.cpu().numpy())
            df_to_mesh(os.path.join(output_path,name+"_target_mesh.ply"), target_df, trunc=1.0, color=np.array([0, 169, 255]))

        if pred_occs is not None:
            pred_occ = preprocess_occ(pred_occs[k], pred=True)
            occ_to_mesh(os.path.join(output_path,name+"_pred_mesh.ply"), pred_occ)

        if target_occs is not None:
            target_occ = preprocess_occ(target_occs[k], pred=False)
            occ_to_mesh(os.path.join(output_path,name+"_target_mesh.ply"), target_occ)



# if __name__ == '__main__':
#     occ_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-voxelized-gt/03001627/1a6f615e8b1b5ae4dbbc9440457e303e__0__.txt"
#     df_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-voxelized-gt/03001627/1a6f615e8b1b5ae4dbbc9440457e303e_occ_2_df.ply"
#     occ_grid = loader.load_sample(occ_file)
#     occ_grid = preprocess_occ(occ_grid, pred=False)
#     occ_to_mesh("/home/parika/WorkingDir/complete3D/Assets/shapenet-voxelized-gt/03001627/1a6f615e8b1b5ae4dbbc9440457e303e.ply", occ_grid)
