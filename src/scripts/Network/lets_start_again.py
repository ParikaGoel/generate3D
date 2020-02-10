import torch
import numpy as np
from PIL import Image
import dataset_loader as loader

# create camera intrinsics
# with these intrinsics, below mapping happens :
# (0, 0, 0.5) -> [-0.2438, -0.2438,  0.5000]
# (511, 511, 0.5) -> [0.2429, 0.2429, 0.5000]
# (0, 0, 1.5) -> [-0.7314, -0.7314,  1.5000]
# (511, 511, 1.5) -> [0.7286, 0.7286, 1.5000]
def make_intrinsic():
    intrinsic = torch.eye(4)
    intrinsic[0][0] = 525.0
    intrinsic[1][1] = 525.0
    intrinsic[0][2] = 512 / 2
    intrinsic[1][2] = 512 / 2
    return intrinsic


# create transformation matrix from world to grid
# each grid coordinate represents the global coord corresponding to its bottom-leftmost corner
# usage :
# grid_coord = torch.matmul(world_to_grid,world_coord).int()
# world = torch.matmul(grid_to_world, grid_coord.float())
def make_world_to_grid():
    world_to_grid = torch.eye(4)
    world_to_grid[0][3] = world_to_grid[1][3] = world_to_grid[2][3] = 0.5
    world_to_grid *= 32
    world_to_grid[3][3] = 1.0
    return world_to_grid


class ProjectionHelper():
    def __init__(self, device, intrinsic, depth_min, depth_max, image_dims, volume_dims, voxel_size):
        self.device = device
        self.intrinsic = intrinsic.to(device)
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.image_dims = image_dims
        self.volume_dims = volume_dims
        self.voxel_size = voxel_size

    def depth_to_skeleton(self, ux, uy, depth):
        x = (ux - self.intrinsic[0][2]) / self.intrinsic[0][0]
        y = (uy - self.intrinsic[1][2]) / self.intrinsic[1][1]
        return torch.Tensor([depth * x, depth * y, depth])

    def skeleton_to_depth(self, p):
        x = (p[0] * self.intrinsic[0][0]) / p[2] + self.intrinsic[0][2]
        y = (p[1] * self.intrinsic[1][1]) / p[2] + self.intrinsic[1][2]
        return torch.Tensor([x, y, p[2]])

    def show_color_projection(self, index_map, flatten_color_grid):
        proj_img = torch.empty((self.image_dims[1], self.image_dims[0], 3), dtype=torch.uint8).fill_(255)

        for u in range(512):
            for v in range(512):
                if index_map[v,u] != 32768:
                    proj_img[v, u, :] = flatten_color_grid[:,index_map[v, u]]

        img_np = proj_img.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img_np)
        img.show()

    def compute_color_projection(self, color_grid, grid_to_world):
        flatten_color_grid = torch.flatten(color_grid, start_dim=1, end_dim=-1)
        occ_mask = torch.lt(flatten_color_grid[0], 256) * torch.lt(flatten_color_grid[1], 256) * torch.lt(
            flatten_color_grid[2], 256)
        occ_mask = torch.flatten(occ_mask, start_dim=0, end_dim=-1)

        lin_ind_volume = torch.arange(0, self.volume_dims[0] * self.volume_dims[1] * self.volume_dims[2],
                                      out=torch.LongTensor()).to(self.device)
        lin_ind_volume = lin_ind_volume[occ_mask]

        # here lin_ind_volume contains the grid coordinates which are occupied
        grid_coords = grid_to_world.new(4, lin_ind_volume.size(0)).int()
        grid_coords[2] = lin_ind_volume / (self.volume_dims[0] * self.volume_dims[1])
        tmp = lin_ind_volume - (grid_coords[2] * self.volume_dims[0] * self.volume_dims[1]).long()
        grid_coords[1] = tmp / self.volume_dims[0]
        grid_coords[0] = torch.remainder(tmp, self.volume_dims[0])
        grid_coords[3].fill_(1)
        # coords contains the occupied grid coordinates : shape (4 x N) N -> number of grid cells occupied

        # transform to current frame
        camera_coords = torch.mm(grid_to_world, grid_coords.float())

        # project into image
        p = camera_coords.clone()
        p[0] = (p[0] * intrinsic[0][0]) / torch.abs(p[2]) + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / torch.abs(p[2]) + intrinsic[1][2]
        p = torch.round(p).long()

        p_range = p.clone()
        p_range[0] = p_range[0] + 32
        p_range[1] = p_range[1] + 32

        p = torch.clamp(p, min=0, max=511)
        p_range = torch.clamp(p_range, min=0, max=511)

        index_map = torch.empty((self.image_dims[1], self.image_dims[0]), dtype=torch.long).fill_(
            self.volume_dims[0] * self.volume_dims[1] * self.volume_dims[2])

        for i in range(lin_ind_volume.size(0)):
            index_map[p[1, i]:p_range[1, i], p[0, i]:p_range[0, i]] = torch.clamp(index_map[p[1, i]:p_range[1, i], p[0, i]:p_range[0, i]], max=lin_ind_volume[i])

        self.show_color_projection(index_map, flatten_color_grid)

        return index_map

    def show_projection(self, img_mask):
        proj_img = torch.empty((self.image_dims[1], self.image_dims[0]), dtype=torch.uint8).fill_(1)
        proj_img[img_mask] = 0

        img_np = proj_img.cpu().numpy().astype(np.uint8)
        img_np[img_np == 1] = 255
        img = Image.fromarray(img_np, 'L')
        img.show()

    def compute_projection(self, occ_grid, grid_to_world):
        flatten_occ_grid = torch.flatten(occ_grid, start_dim=0, end_dim=-1)
        occ_mask = torch.eq(flatten_occ_grid, 1)

        lin_ind_volume = torch.arange(0, self.volume_dims[0] * self.volume_dims[1] * self.volume_dims[2],
                                      out=torch.LongTensor()).to(self.device)
        lin_ind_volume = lin_ind_volume[occ_mask]

        # here lin_ind_volume contains the grid coordinates which are occupied
        grid_coords = grid_to_world.new(4, lin_ind_volume.size(0)).int()
        grid_coords[2] = lin_ind_volume / (self.volume_dims[0] * self.volume_dims[1])
        tmp = lin_ind_volume - (grid_coords[2] * self.volume_dims[0] * self.volume_dims[1]).long()
        grid_coords[1] = tmp / self.volume_dims[0]
        grid_coords[0] = torch.remainder(tmp, self.volume_dims[0])
        grid_coords[3].fill_(1)
        # coords contains the occupied grid coordinates : shape (4 x N) N -> number of grid cells occupied

        # transform to current frame
        camera_coords = torch.mm(grid_to_world, grid_coords.float())

        # project into image
        p = camera_coords.clone()
        p[:, 2] = p[:, 2] - 1.0
        p[0] = (p[0] * intrinsic[0][0]) / torch.abs(p[2]) + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / torch.abs(p[2]) + intrinsic[1][2]
        p = torch.round(p).long()

        p_range = p.clone()
        p_range[0] = p_range[0] + 32
        p_range[1] = p_range[1] + 32

        p = torch.clamp(p, min=0, max=511)
        p_range = torch.clamp(p_range, min=0, max=511)

        index_map = torch.empty((self.image_dims[1], self.image_dims[0]), dtype=torch.long).fill_(
            self.volume_dims[0] * self.volume_dims[1] * self.volume_dims[2])

        for i in range(lin_ind_volume.size(0)):
            index_map[p[1, i]:p_range[1, i], p[0, i]:p_range[0, i]] = torch.clamp(index_map[p[1, i]:p_range[1, i], p[0, i]:p_range[0, i]], max=lin_ind_volume[i])

        img_mask = torch.lt(index_map, self.volume_dims[0] * self.volume_dims[1] * self.volume_dims[2])
        self.show_projection(img_mask)

        index_map = torch.flatten(index_map)
        return index_map


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rainbow_file = "/media/sda2/shapenet/test/rainbow_0_.txt"
    gt_file = "/media/sda2/shapenet/test/fd013bea1e1ffb27c31c70b1ddc95e3f__test__.txt"
    intrinsic = make_intrinsic()
    world_to_grid = make_world_to_grid().to(device)
    grid_to_world = torch.inverse(world_to_grid)
    voxel_size = 1 / 32
    rainbow_occ = loader.load_color_sample(rainbow_file).to(device)
    gt_occ = loader.load_sample(gt_file).to(device)

    projection_helper = ProjectionHelper(device, intrinsic, 0.5, 1.5, [512, 512], [32, 32, 32], voxel_size)
    # projection_helper.compute_color_projection(rainbow_occ, grid_to_world)
    lin_index_map = projection_helper.compute_projection(gt_occ, grid_to_world)

