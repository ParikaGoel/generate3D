import numpy as np
from config import *


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
        # Since we are looking in the -ve z direction, to have the z max bound, we need to subtract
        self._max_bound[2] = self._max_bound[2] - 2 * grid_size
        self._occ_grid = np.zeros((dim, dim, dim)).astype(int)
        # last dimension in the color grid is the number of channels e.g RGB -> 3
        self._color_grid = np.zeros((dim, dim, dim, 3)).astype(int)

    # each voxel represents the global coordinate corresponding to its center
    def get_global_coord(self, grid_coord):
        global_coord = np.copy(grid_coord)
        global_coord = global_coord.astype(float)
        global_coord += 0.5
        global_coord *= self._voxel_scale
        # we need to change the sign of z-coordinate since we are looking in the -ve z-direction
        global_coord[2] = -global_coord[2]
        global_coord += self._min_bound
        return global_coord

    # Returns the grid coord corresponding to global coordinate
    # If global coordinate does not lie inside the grid, returns None
    def get_grid_coord(self, global_coord):

        if not self.contains_global_coord(global_coord):
            return None

        grid_coord = np.copy(global_coord)
        grid_coord -= self._min_bound
        grid_coord[2] = -grid_coord[2]
        grid_coord /= self._voxel_scale

        grid_coord = grid_coord.astype(int)

        if np.any(grid_coord > self._dim - 1):
            return None

        return grid_coord

    def contains_global_coord(self, global_coord):

        # since we are looking in the -ve z-direction, max bound for z will be lesser than min bound for z
        if self._min_bound[0] <= global_coord[0] <= self._max_bound[0] and \
                self._min_bound[1] <= global_coord[1] <= self._max_bound[1] and \
                self._max_bound[2] <= global_coord[2] <= self._min_bound[2]:
            return True
        else:
            return False

    @property
    def min_bound(self):
        return self._min_bound

    @property
    def max_bound(self):
        return self._max_bound

    @property
    def voxel_scale(self):
        return self._voxel_scale

    @property
    def occ_grid(self):
        return self._occ_grid

    @occ_grid.setter
    def occ_grid(self, value):
        self._occ_grid = value

    @property
    def color_grid(self):
        return self._color_grid

    @color_grid.setter
    def color_grid(self, value):
        self._color_grid = value

    def is_occupied(self, grid_coord):
        if self._occ_grid[grid_coord[0], grid_coord[1], grid_coord[2]] == 1:
            return True
        return False

    def set_occupancy(self, grid_coord, is_occupied=1):
        self._occ_grid[grid_coord[0], grid_coord[1], grid_coord[2]] = is_occupied

    def set_color(self, grid_coord, color):
        self.set_occupancy(grid_coord, is_occupied=1)
        self._color_grid[grid_coord[0], grid_coord[1], grid_coord[2], :] = color
        # if np.all(self._color_grid[grid_coord[0], grid_coord[1], grid_coord[2], :] == 0):
        #     self._color_grid[grid_coord[0], grid_coord[1], grid_coord[2], :] = color
        # else:
        #     self._color_grid[grid_coord[0], grid_coord[1], grid_coord[2], :] += color
        #     self._color_grid[grid_coord[0], grid_coord[1], grid_coord[2], :] //= 2

    # Saves the color grid of the voxel in a file
    # Occupancy grid is implicitly saved
    def save_vox(self, filename):
        positions = np.where(self._occ_grid == 1)
        with open(filename, "w") as f:
            for i, j, k in zip(*positions):
                color = self._color_grid[i, j, k]
                data = np.column_stack((i, j, k, color[0], color[1], color[2]))
                np.savetxt(f, data, fmt='%d %d %d %d %d %d', delimiter=' ')

    def load_vox(self, filename):
        voxel = np.loadtxt(filename,dtype=int)
        self._occ_grid = np.zeros((32, 32, 32), dtype=int)
        self._color_grid = np.zeros((32, 32, 32, 3), dtype=int)
        for data in voxel:
            grid_coord = np.array((data[0], data[1], data[2])).astype(int)
            color = np.array((data[3], data[4], data[5])).astype(int)
            self.set_color(grid_coord, color)
            self.set_occupancy(grid_coord, is_occupied=1)

    def to_mesh(self, filename, transform=None):
        cube_verts = np.array([[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
                               [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]])  # 8 points

        cube_faces = np.array([[0, 1, 2], [2, 3, 0], [1, 5, 6], [6, 2, 1], [7, 6, 5], [5, 4, 7],
                               [4, 0, 3], [3, 7, 4], [4, 5, 1], [1, 0, 4], [3, 2, 6],
                               [6, 7, 3]])  # 6 faces (12 triangles)

        verts = []
        faces = []
        curr_vertex = 0

        positions = np.where(self._occ_grid == 1)
        min_bound = np.array([-0.5, -0.5, -0.5])
        for i, j, k in zip(*positions):
            for cube_vert in cube_verts:
                vertex = (cube_vert * 0.45 + np.array([i, j, k])).astype(float)
                vertex *= self._voxel_scale
                # since we are looking in -ve z direction
                # vertex[2] = -vertex[2]
                vertex += min_bound

                if transform is not None:
                    rotation = transform[0:3, 0:3]
                    translation = transform[0:3, 3]
                    vertex = np.matmul(rotation, vertex) + translation
                color = self._color_grid[i, j, k, :]
                # color = [178, 120, 33]
                vertex = np.append(vertex, color)
                vertex = list(vertex)
                verts.append(vertex)

            for cube_face in cube_faces:
                face = curr_vertex + cube_face
                faces.append(list(face))

            curr_vertex += len(cube_verts)

        write_ply(filename, verts, faces)


def create_voxel_grid(cam):
    grid_size = abs(cam.z_far - cam.z_near)
    voxel_min_bound = np.array([-grid_size / 2, -grid_size / 2, -cam.z_near])
    voxel_dim = vox_dim
    voxel_grid = VoxelGrid(voxel_min_bound, voxel_dim, grid_size)
    print("Min bound: ", voxel_grid.min_bound)
    print("Max bound: ", voxel_grid.max_bound)
    print("Voxel Scale: ", voxel_grid.voxel_scale)
    return voxel_grid

def txt_to_mesh(txt_file, ply_file, grid_size = None):
    cube_verts = np.array([[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
                           [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0],[-1.0, 1.0, -1.0]])  # 8 points

    cube_faces = np.array([[0, 1, 2], [2, 3, 0], [1, 5, 6], [6, 2, 1], [7, 6, 5], [5, 4, 7],
                           [4, 0, 3], [3, 7, 4], [4, 5, 1], [1, 0, 4], [3, 2, 6], [6, 7, 3]])  # 6 faces (12 triangles)

    verts = []
    faces = []
    curr_vertex = 0

    if grid_size is None:
        grid_size = 1

    min_bound = np.array([-grid_size/2,-grid_size/2,-grid_size/2])
    voxel_scale = grid_size / vox_dim

    voxel = np.loadtxt(txt_file, dtype=int)
    for data in voxel:
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


if __name__ == '__main__':
    txt_file = "/media/sda2/shapenet/test/rainbow_0_.txt"
    ply_file = "/media/sda2/shapenet/test/rainbow_0_.ply"
    txt_to_mesh(txt_file, ply_file)