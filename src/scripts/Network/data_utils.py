import sys
sys.path.append('../.')
import os
import torch
import config
import numpy as np
from data_formats import write_ply


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


def save_predictions(output_path, names, pred_dfs, target_dfs):
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

