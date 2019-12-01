import numpy as np


class VoxelGrid:

    # origin : bottom-leftmost point of the voxel grid
    # dim : dimensions of the voxel grid
    # grid_size : units in the world system voxel grid represents
    # voxel_scale : units in the world system each voxel represents
    def __init__(self, origin, dim=32, grid_size=512):
        self._dim = dim
        self._voxel_scale = grid_size / dim
        self._grid_size = grid_size
        self._min_bound = origin
        self._max_bound = origin + grid_size
        self._occ_grid = np.zeros((dim, dim, dim)).astype(int)
        self._color_grid = np.ones((dim, dim, dim))

    # each voxel represents the global coordinate corresponding to its center
    def get_global_coord(self, grid_coord):
        global_coord = np.copy(grid_coord)
        global_coord = global_coord.astype(float)
        global_coord += 0.5
        global_coord *= self._voxel_scale
        global_coord += self._min_bound
        return global_coord

    # Returns the grid coord cooresponding to global coordinate
    # If global coordinate does not lie inside the grid, returns None
    def get_grid_coord(self, global_coord):

        if not self.contains_global_coord(global_coord):
            return None

        grid_coord = np.copy(global_coord)
        grid_coord -= self._min_bound
        grid_coord /= self._voxel_scale
        grid_coord -= 0.5
        grid_coord = grid_coord.astype(int)
        return grid_coord

    def contains_global_coord(self, global_coord):

        if self._min_bound[0] <= global_coord[0] <= self._max_bound[0] and \
                self._min_bound[1] <= global_coord[1] <= self._max_bound[1] and \
                self._min_bound[2] <= global_coord[2] <= self._max_bound[2]:
            return True
        else:
            return False

    def get_min_bound(self):
        return self._min_bound

    def get_max_bound(self):
        return self._max_bound

    def get_voxel_scale(self):
        return self._voxel_scale

    def get_occupancy_vals(self):
        return self._occ_grid

    def is_occupied(self, grid_coord):
        if self._occ_grid[grid_coord[0], grid_coord[1], grid_coord[2]] == 1:
            return True
        return False

    def set_occupancy(self, grid_coord, is_occupied=1):
        self._occ_grid[grid_coord[0], grid_coord[1], grid_coord[2]] = is_occupied

    def set_color(self, grid_coord, color):
        self._occ_grid[grid_coord[0], grid_coord[1], grid_coord[2]] = color

    def save_as_ply(self, filename):
        cube_verts = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
                      [1, 1, 1]])  # 8 points

        cube_faces = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 6], [3, 7, 6], [0, 2, 6], [0, 6, 4], [0, 5, 1],
                      [0, 4, 5], [6, 7, 5], [6, 5, 4], [1, 7, 3], [1, 5, 7]])  # 6 faces (12 triangles)

        verts = []
        faces = []
        curr_vertex = 0

        positions = np.where(self._occ_grid == 1)
        for i, j, k in zip(*positions):
            for cube_vert in cube_verts:
                vertex = (cube_vert + np.array([i, j, k])).astype(float)
                vertex *= self._voxel_scale
                vertex += self._min_bound
                vertex /= self._grid_size
                vertex = np.append(vertex, [0, 169, 255])
                vertex = list(vertex)
                verts.append(vertex)

            for cube_face in cube_faces:
                face = curr_vertex + cube_face
                faces.append(list(face))

            curr_vertex += len(cube_verts)

        write_ply(filename, verts, faces)


def write_ply(filename, verts, faces):
    file = open(filename, "w")
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
