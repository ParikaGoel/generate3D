import trimesh
from mesh_to_sdf import mesh_to_voxels
import numpy as np
from Network import dataset_loader as loader
import voxel_grid
import torch


if __name__=='__main__':
    obj_file = "./Assets_remote/shapenet-data/04379243/1a00aa6b75362cc5b324368d54a7416f/models/model_normalized.obj"
    mesh = trimesh.load(obj_file)

    if isinstance(mesh, trimesh.Scene):
        meshes = mesh.dump().sum()
    else:
        meshes = mesh

    centroid = meshes.bounding_box.centroid
    extent = meshes.bounding_box.extents

    voxels = mesh_to_voxels(mesh, 32, pad=False)

    file = "./Assets/test"
    np.save(file, voxels)

    trunc_dist = 2/32
    # loader.sdf_to_mesh(file=sdf_file+".npy", trunc_dist=voxel_size)
    sdf_grid = torch.from_numpy(np.load(file+".npy"))
    mask = torch.gt(sdf_grid, -trunc_dist) & torch.lt(sdf_grid, trunc_dist)
    positions = np.where(mask.cpu().numpy())

    grid_lst = []
    color = np.array([169, 0, 255])

    for i, j, k in zip(*positions):
        data = np.array((i, j, k, color[0], color[1], color[2]))
        grid_lst.append(data)

    verts, faces = voxel_grid.grid_to_mesh(grid_lst, file+".ply", 2)
    new_verts = []

    for data in verts:
        coord = np.array((data[0], data[1], data[2]))
        color = np.array((data[3], data[4], data[5]))

        coord *= np.max(extent) / 2
        coord += centroid

        vertex = np.append(coord, color)
        vertex = list(vertex)
        new_verts.append(vertex)

    voxel_grid.write_ply(file+".ply", new_verts, faces)

