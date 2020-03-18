import torch
import trimesh
import voxel_grid
import numpy as np
from mesh_to_sdf import mesh_to_voxels
from Network import dataset_loader as loader


if __name__=='__main__':
    sdf_file = "../../Assets/shapenet-voxelized-gt/03001627/1a6f615e8b1b5ae4dbbc9440457e303e.npy"
    df_file = "../../Assets/shapenet-voxelized-gt/03001627/1a6f615e8b1b5ae4dbbc9440457e303e__0__.df"
    obj_file = "../../Assets/shapenet-data/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/models/model_normalized.obj"

    mesh = trimesh.load(obj_file)
    translate = mesh.bounding_box.centroid
    scale = 2 / np.max(mesh.bounding_box.extents)

    # voxels = mesh_to_voxels(mesh, 32, pad=False)

    df_grid = loader.load_df(df_file)[0].numpy()
    trunc_dist = 1/32
    sdf_grid = np.load(sdf_file)
    sdf_grid = torch.from_numpy(sdf_grid)
    mask = torch.gt(sdf_grid, -trunc_dist) & torch.lt(sdf_grid, trunc_dist)
    positions = np.where(mask.cpu().numpy())

    grid_lst = []
    color = np.array([169, 0, 255])

    for i, j, k in zip(*positions):
        data = np.array((i, j, k, color[0], color[1], color[2]))
        grid_lst.append(data)

    ply_file = sdf_file[:sdf_file.rfind('.')] + '_sdf_.ply'
    verts, faces = voxel_grid.grid_to_mesh(grid_lst, ply_file, 2)

    new_verts = []

    for data in verts:
        coord = np.array((data[0], data[1], data[2]))
        color = np.array((data[3], data[4], data[5]))
        coord = coord / scale
        coord = coord + translate
        coord = list(np.append(coord, color))
        new_verts.append(coord)

    voxel_grid.write_ply(ply_file, new_verts, faces)
