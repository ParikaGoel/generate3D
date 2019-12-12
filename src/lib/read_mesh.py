import open3d as o3d


def main():
    mesh = o3d.io.read_triangle_mesh("/home/parika/WorkingDir/complete3D/results/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/raycasted_voxel/20191201230817.ply")
    # o3d.visualization.draw_geometries([mesh])
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, 0.01)
    o3d.visualization.draw_geometries([voxel_grid])
    

if __name__ == '__main__':
    main()