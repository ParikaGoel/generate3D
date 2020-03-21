import torch
import numpy as np


def generate_ply(grid_positions, ply_filename, grid_size=None):
    cube_verts = np.array([[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
    [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]]) # 8 points

    cube_faces = np.array([[0, 1, 2], [2, 3, 0], [1, 5, 6], [6, 2, 1], [7, 6, 5], [5, 4, 7],
                           [4, 0, 3], [3, 7, 4], [4, 5, 1], [1, 0, 4], [3, 2, 6], [6, 7, 3]])  # 6 faces (12 triangles)

    verts = []
    faces = []
    curr_vertex = 0

    if grid_size is None:
        grid_size = 1

    min_bound = np.array([-grid_size / 2, -grid_size / 2, -grid_size / 2])
    voxel_scale = grid_size / 64

    color = np.array([169, 0, 255])

    for i, j, k in zip(*grid_positions):
        for cube_vert in cube_verts:
            vertex = (cube_vert * 0.45 + np.array([i, j, k])).astype(float)
            vertex *= voxel_scale
            vertex += min_bound
            vertex = np.append(vertex, color)
            vertex = list(vertex)
            verts.append(vertex)

        for cube_face in cube_faces:
            face = curr_vertex + cube_face
            faces.append(list(face))

        curr_vertex += len(cube_verts)

        file = open(ply_filename, "w")
        file.write('''ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            element face %d
            property list uchar int vertex_index
            end_header
            ''' % (len(verts), len(faces)))

        for vert in verts:
            file.write('%f %f %f %d %d %d\n' % tuple(vert))

        for face in faces:
            file.write('3 %d %d %d\n' % tuple(face))

        file.close()


if __name__=='__main__':
    sdf_file = "../../Assets_remote/output-network/Net2/04379243/predicted_test_output/sdf/best_model.npy"
    ply_file = "../../Assets_remote/output-network/Net2/04379243/predicted_test_output/sdf/best_model.ply"
    trunc_dist = 1.0
    sdf_grid = torch.from_numpy(np.load(sdf_file))
    mask = torch.ge(sdf_grid, -trunc_dist) & torch.le(sdf_grid, trunc_dist)
    positions = np.where(mask.cpu().numpy())
    generate_ply(positions, ply_file, grid_size=2)
