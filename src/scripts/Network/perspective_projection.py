import sys

sys.path.append('../.')
import torch
import config
import losses
import numpy as np
from PIL import Image
import dataset_loader as loader
from torch.autograd import Function


# create camera intrinsics
# with these intrinsics, below mapping happens :
# (0, 0, 0.5) -> [-0.2438, -0.2438,  0.5000]
# (511, 511, 0.5) -> [0.2429, 0.2429, 0.5000]
# (0, 0, 1.5) -> [-0.7314, -0.7314,  1.5000]
# (511, 511, 1.5) -> [0.7286, 0.7286, 1.5000]
def make_intrinsic():
    intrinsic = torch.eye(4)
    intrinsic[0][0] = config.focal
    intrinsic[1][1] = config.focal
    intrinsic[0][2] = config.render_img_width / 2
    intrinsic[1][2] = config.render_img_height / 2
    return intrinsic


# create transformation matrix from world to grid
# each grid coordinate represents the global coord corresponding to its bottom-leftmost corner
# usage :
# grid_coord = torch.matmul(world_to_grid,world_coord).int()
# world = torch.matmul(grid_to_world, grid_coord.float())
def make_world_to_grid():
    world_to_grid = torch.eye(4)
    world_to_grid[0][3] = world_to_grid[1][3] = world_to_grid[2][3] = 0.5
    world_to_grid *= config.vox_dim
    world_to_grid[3][3] = 1.0
    return world_to_grid


class ProjectionHelper:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grid_to_world = torch.inverse(make_world_to_grid()).to(self.device)
        self.intrinsic = make_intrinsic().to(self.device)
        self.depth_min = config.znear
        self.depth_max = config.zfar
        self.image_dims = [config.render_img_width, config.render_img_height]
        self.volume_dims = [config.vox_dim, config.vox_dim, config.vox_dim]
        self.voxel_size = 1 / config.vox_dim

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
                if index_map[v, u] != 32768:
                    proj_img[v, u, :] = flatten_color_grid[:, index_map[v, u]]

        img_np = proj_img.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img_np)
        img.show()

    def compute_color_projection(self, color_grid, world_to_camera):
        flatten_color_grid = torch.flatten(color_grid, start_dim=1, end_dim=-1)
        occ_mask = torch.lt(flatten_color_grid[0], 256) * torch.lt(flatten_color_grid[1], 256) * torch.lt(
            flatten_color_grid[2], 256)
        occ_mask = torch.flatten(occ_mask, start_dim=0, end_dim=-1)

        lin_ind_volume = torch.arange(0, self.volume_dims[0] * self.volume_dims[1] * self.volume_dims[2],
                                      out=torch.LongTensor()).to(self.device)
        lin_ind_volume = lin_ind_volume[occ_mask]

        # here lin_ind_volume contains the grid coordinates which are occupied
        grid_coords = self.grid_to_world.new(4, lin_ind_volume.size(0)).int()
        grid_coords[2] = lin_ind_volume / (self.volume_dims[0] * self.volume_dims[1])
        tmp = lin_ind_volume - (grid_coords[2] * self.volume_dims[0] * self.volume_dims[1]).long()
        grid_coords[1] = tmp / self.volume_dims[0]
        grid_coords[0] = torch.remainder(tmp, self.volume_dims[0])
        grid_coords[3].fill_(1)
        # coords contains the occupied grid coordinates : shape (4 x N) N -> number of grid cells occupied

        # cube vertex coords
        vertex_coords = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1],
                                      [0, 0, 1, 1, 0, 0, 1, 1],
                                      [0, 0, 0, 0, 1, 1, 1, 1],
                                      [0, 0, 0, 0, 0, 0, 0, 0]]).to(self.device)

        corners = grid_coords.unsqueeze(0).repeat(8, 1, 1)
        for i in range(8):
            corners[i] = corners[i] + vertex_coords[:, i, None]

        # transform to current frame
        camera_coords = torch.matmul(world_to_camera, torch.matmul(self.grid_to_world, corners.float()))

        # project into image
        p = camera_coords.clone()
        for i in range(8):
            p[i, 1, :] = -p[i, 1, :]  # perspective projection flips the image in y- direction
            p[i, 0] = (p[i, 0] * self.intrinsic[0][0]) / torch.abs(p[i, 2]) + self.intrinsic[0][2]
            p[i, 1] = (p[i, 1] * self.intrinsic[1][1]) / torch.abs(p[i, 2]) + self.intrinsic[1][2]
        p = torch.round(p).long()

        p = torch.clamp(p, min=0, max=511)
        pmin = torch.min(p, dim=0).values
        pmax = torch.max(p, dim=0).values

        # # Note : to be used when camera is seeing in the +ve z-direction
        # index_map = torch.empty((self.image_dims[1], self.image_dims[0]), dtype=torch.long).fill_(32 * 32 * 32)
        #
        # for i in range(lin_ind_volume.size(0)):
        #     index_map[p[1, i]:p_range[1, i], p[0, i]:p_range[0, i]] = torch.clamp(index_map[p[1, i]:p_range[1, i], p[0, i]:p_range[0, i]], max=lin_ind_volume[i])

        # Note : to be used when camera is seeing in the -ve z-direction
        index_map = torch.empty((self.image_dims[1], self.image_dims[0]), dtype=torch.long).fill_(0)

        for i in range(lin_ind_volume.size(0)):
            index_map[pmin[1, i]:pmax[1, i], pmin[0, i]:pmax[0, i]] = torch.clamp(
                index_map[pmin[1, i]:pmax[1, i], pmin[0, i]:pmax[0, i]], min=lin_ind_volume[i])

        self.show_color_projection(index_map, flatten_color_grid)

        return index_map

    def show_projection(self, index_map):
        img_mask = torch.gt(index_map, 0)
        proj_img = torch.empty((self.image_dims[1] * self.image_dims[0]), dtype=torch.uint8).fill_(1)
        proj_img[img_mask] = 0
        proj_img = torch.reshape(proj_img, (self.image_dims[1], self.image_dims[0]))

        img_np = proj_img.cpu().numpy().astype(np.uint8)
        img_np[img_np == 1] = 255
        img = Image.fromarray(img_np, 'L')
        img.show()

    def compute_projection(self, occ_grid, world_to_camera):
        occ_grid = occ_grid[0]  # removes the channel dimension
        flatten_occ = torch.flatten(occ_grid, start_dim=0, end_dim=-1)
        occ_mask = torch.eq(flatten_occ, 1.0)

        lin_ind_volume = torch.arange(0, self.volume_dims[0] * self.volume_dims[1] * self.volume_dims[2],
                                      out=torch.LongTensor()).to(self.device)
        lin_ind_volume = lin_ind_volume[occ_mask]

        # here lin_ind_volume contains the grid coordinates which are occupied
        grid_coords = self.grid_to_world.new(4, lin_ind_volume.size(0)).int()
        grid_coords[2] = lin_ind_volume / (self.volume_dims[0] * self.volume_dims[1])
        tmp = lin_ind_volume - (grid_coords[2] * self.volume_dims[0] * self.volume_dims[1]).long()
        grid_coords[1] = tmp / self.volume_dims[0]
        grid_coords[0] = torch.remainder(tmp, self.volume_dims[0])
        grid_coords[3].fill_(1)
        # coords contains the occupied grid coordinates : shape (4 x N) N -> number of grid cells occupied

        # cube vertex coords
        vertex_coords = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1],
                                      [0, 0, 1, 1, 0, 0, 1, 1],
                                      [0, 0, 0, 0, 1, 1, 1, 1],
                                      [0, 0, 0, 0, 0, 0, 0, 0]]).to(self.device)

        corners = grid_coords.unsqueeze(0).repeat(8, 1, 1)
        for i in range(8):
            corners[i] = corners[i] + vertex_coords[:, i, None]

        # transform to current frame
        camera_coords = torch.matmul(world_to_camera, torch.matmul(self.grid_to_world, corners.float()))

        # project into image
        p = camera_coords.clone()
        for i in range(8):
            p[i, 1, :] = -p[i, 1, :]  # perspective projection flips the image in y- direction
            p[i, 0] = (p[i, 0] * self.intrinsic[0][0]) / torch.abs(p[i, 2]) + self.intrinsic[0][2]
            p[i, 1] = (p[i, 1] * self.intrinsic[1][1]) / torch.abs(p[i, 2]) + self.intrinsic[1][2]
        p = torch.round(p).long()

        p = torch.clamp(p, min=0, max=511)
        pmin = torch.min(p, dim=0).values
        pmax = torch.max(p, dim=0).values

        # Note : to be used when camera is seeing in the -ve z-direction
        index_map = torch.empty((self.image_dims[1], self.image_dims[0]), dtype=torch.long).fill_(-1)

        for i in range(lin_ind_volume.size(0)):
            index_map[pmin[1, i]:pmax[1, i], pmin[0, i]:pmax[0, i]] = torch.clamp(
                index_map[pmin[1, i]:pmax[1, i], pmin[0, i]:pmax[0, i]], min=lin_ind_volume[i])

        # self.show_projection(index_map)
        index_map = torch.flatten(index_map)

        return index_map

    def project_occ_n_views(self, occ_grid, poses):
        lin_index_map = torch.stack([self.compute_projection(occ_grid, pose) for pose in poses])
        return lin_index_map

    def project_batch_n_views(self, occ_batch, poses_batch):
        batch_size = occ_batch.size(0)

        batch_index_map = torch.stack([self.project_occ_n_views(occ_batch[i],poses_batch[i])
                                      for i in range(batch_size)])

        return batch_index_map

    def forward(self, occ_grid, lin_index_map):
        """
        
        :param inp_occ_grid: occupancy grid for which viewpoint mappings are given
        :param lin_index_map: index mapping for every viewpoint ; shape - (N x (32*32*32))
        :return: 
        """
        batch_size = occ_grid.size(0)
        occ_grid = torch.flatten(occ_grid, start_dim=1, end_dim=-1)
        invalid_col = occ_grid.new_empty((batch_size, 1)).fill_(-1.0)
        occ_grid = torch.cat([occ_grid, invalid_col], dim=1)  # Add -1 to the end for invalid value

        proj_imgs = torch.stack([occ_grid[i][lin_index_map[i]] for i in range(batch_size)])
        mask = torch.eq(proj_imgs, -1)
        proj_imgs[mask] = 0

        return proj_imgs

    def backward(self, grad_output, lin_index_map, inp_occ_grid):
        inp_occ_grid = inp_occ_grid[0]  # removes the channel dimension
        flatten_occ = torch.flatten(inp_occ_grid, start_dim=0, end_dim=-1).to(self.device)
        flatten_occ = torch.cat(
            [flatten_occ, torch.tensor([-1.0]).to(self.device)])  # Add -1 to the end for invalid value

        n_views = lin_index_map.size(0)
        occs = []
        for i in range(n_views):
            new_occ = flatten_occ.clone()
            new_occ[lin_index_map[i]] = grad_output[i]
            occs.append(new_occ)

        occs = torch.stack(occs)[:, :32768]
        occ_grid = torch.reshape(torch.mean(occs, dim=0), (32, 32, 32)).unsqueeze(0)
        return occ_grid


# Inherit from Function
class Projection(Function):

    @staticmethod
    def forward(ctx, occ_grid, lin_index_map):
        batch_size = occ_grid.size(0)
        occ_grid = torch.flatten(occ_grid, start_dim=1, end_dim=-1)
        ctx.save_for_backward(lin_index_map, occ_grid)
        invalid_col = occ_grid.new_empty((batch_size, 1)).fill_(-1.0)
        occ_grid = torch.cat([occ_grid, invalid_col], dim=1)  # Add -1 to the end for invalid value

        proj_imgs = torch.stack([occ_grid[i][lin_index_map[i]] for i in range(batch_size)])
        mask = torch.eq(proj_imgs, -1)
        proj_imgs[mask] = 0

        return proj_imgs

    @staticmethod
    def backward(ctx, grad_output):
        lin_index_map, flatten_occ = ctx.saved_variables
        batch_size = flatten_occ.size(0)
        occ_size = flatten_occ.size(1)
        # invalid_col = flatten_occ.new_empty((batch_size, 1)).fill_(-1.0)
        # flatten_occ = torch.cat([flatten_occ, invalid_col], dim=1)  # Add -1 to the end for invalid value

        # ToDo(Parika) : Need to take the average of all the pixel values corresponding to same voxel

        occ_grids = []
        for b in range(batch_size):
            occ = flatten_occ[b]
            index_map = lin_index_map[b]
            n_views = index_map.size(0)
            mask = torch.eq(index_map, -1)
            index_map = torch.stack([torch.add(index_map[i, :], i * occ_size) for i in range(index_map.size(0))])
            index_map[mask] = -1
            index_map = torch.flatten(index_map)

            grad = grad_output[b]
            tmp = torch.cat([occ.repeat(n_views), torch.tensor([-1])]).float()
            tmp[index_map] = grad
            tmp = torch.reshape(torch.mean(tmp[:-1].reshape((-1, occ_size)), dim=0), (1, 32, 32, 32))
            occ_grids.append(tmp)

        batch_occ = torch.stack(occ_grids)
        return batch_occ, None


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rainbow_file = "../../../Assets_remote/test/rainbow_frustrum_.txt"
    gt_file = "../../../Assets_remote/test/fd013bea1e1ffb27c31c70b1ddc95e3f__test__.txt"
    cam_file = "../../../Assets_remote/test/fd013bea1e1ffb27c31c70b1ddc95e3f/pose/pose16.json"
    gt_img_file = "../../../Assets_remote/test/fd013bea1e1ffb27c31c70b1ddc95e3f/color/color00.png"
    rainbow_occ = loader.load_color_sample(rainbow_file).to(device)
    gt_occ = loader.load_sample(gt_file).to(device)
    gt_img = loader.load_img(gt_img_file).to(device)
    world_to_camera = loader.load_pose(cam_file).to(device)
    world_to_camera = torch.reshape(world_to_camera, (4, 4))
    world_to_camera[:, 3] = torch.tensor([0.0, 0.0, -1.2, 0.0]).to(device)

    projection_helper = ProjectionHelper()
    # projection_helper.compute_color_projection(test_color_grid, world_to_camera)
    lin_index_map = projection_helper.compute_projection(gt_occ, world_to_camera)
    proj_img = projection_helper.forward(gt_occ, lin_index_map)
    occ_grid = projection_helper.backward(proj_img, lin_index_map, gt_occ)
    proj_loss = losses.vol_proj_loss(proj_img.unsqueeze(0).float(),
                                     gt_img.unsqueeze(0).float(), 1, device)
    print(proj_loss)
    projection_helper.show_projection(proj_img)
