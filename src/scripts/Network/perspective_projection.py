import torch
import losses
import numpy as np
from PIL import Image
import dataset_loader as loader
from torch.autograd import Function


# create camera intrinsics
def make_intrinsic():
    intrinsic = torch.eye(4)
    intrinsic[0][0] = 525.0
    intrinsic[1][1] = 525.0
    intrinsic[0][2] = 512 / 2
    intrinsic[1][2] = 512 / 2
    return intrinsic


# create transformation matrix from world to grid
def make_world_to_grid():
    world_to_grid = torch.eye(4)
    world_to_grid[0][3] = world_to_grid[1][3] = world_to_grid[2][3] = 0.5
    world_to_grid *= 32
    world_to_grid[3][3] = 1.0
    return world_to_grid


class ProjectionHelper():
    def __init__(self, device, intrinsic, depth_min, depth_max, image_dims, volume_dims, voxel_size):
        self.intrinsic = intrinsic
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.image_dims = image_dims
        self.volume_dims = volume_dims
        self.voxel_size = voxel_size
        self.device = device

    # z-coord to be taken negative, bcoz we r assuming our camera to be seeing in -ve z- direction
    def depth_to_skeleton(self, ux, uy, depth):
        x = (ux - self.intrinsic[0][2]) / self.intrinsic[0][0]
        y = (uy - self.intrinsic[1][2]) / self.intrinsic[1][1]
        return torch.Tensor([depth * x, depth * y, -depth])

    def show_projection(self, img_mask):
        proj_img = torch.empty((self.image_dims[1], self.image_dims[0]), dtype=torch.uint8).fill_(1)
        proj_img[img_mask] = 0

        img_np = proj_img.cpu().numpy().astype(np.uint8)
        img_np[img_np == 1] = 255
        img = Image.fromarray(img_np, 'L')
        img.show()

    def compute_projection(self, occ_grid, camera_to_world, world_to_grid):
        occ_grid = occ_grid[0]  # removes the channel dimension

        world_to_camera = torch.inverse(camera_to_world)
        # world_to_camera = camera_to_world
        world_to_camera[2,3] = -1.2
        grid_to_world = torch.inverse(world_to_grid)

        flatten_occ = torch.flatten(occ_grid, start_dim=0, end_dim=-1)
        occ_mask = torch.eq(flatten_occ, 1.0)

        if not occ_mask.any():
            print('error: nothing in occupancy grid')
            return None

        lin_ind_volume = torch.arange(0, self.volume_dims[0] * self.volume_dims[1] * self.volume_dims[2],
                                      out=torch.LongTensor()).to(self.device)
        lin_ind_volume = lin_ind_volume[occ_mask]

        # here lin_ind_volume contains the grid coordinates which are occupied
        grid_coords = camera_to_world.new(4, lin_ind_volume.size(0))
        grid_coords[2] = lin_ind_volume / (self.volume_dims[0] * self.volume_dims[1])
        tmp = lin_ind_volume - (grid_coords[2] * self.volume_dims[0] * self.volume_dims[1]).long()
        grid_coords[1] = tmp / self.volume_dims[0]
        grid_coords[0] = torch.remainder(tmp, self.volume_dims[0])
        grid_coords[3].fill_(1)
        # coords contains the occupied grid coordinates : shape (4 x N) N -> number of grid cells occupied

        grid_coords_max = grid_coords + torch.tensor([-0.5, -0.5, -0.5, 0.0]).to(self.device)[:, None]
        grid_coords_min = grid_coords + torch.tensor([0.5, 0.5, 0.5, 0.0]).to(self.device)[:, None]

        # transform to current frame
        pmax = torch.mm(world_to_camera, torch.mm(grid_to_world, grid_coords_max))
        pmin = torch.mm(world_to_camera, torch.mm(grid_to_world, grid_coords_min))

        # project into image
        pmax[0] = (pmax[0] * self.intrinsic[0][0]) / pmax[2] + self.intrinsic[0][2]
        pmax[1] = (pmax[1] * self.intrinsic[1][1]) / pmax[2] + self.intrinsic[1][2]
        pmax = torch.round(pmax).long()

        pmin[0] = (pmin[0] * self.intrinsic[0][0]) / pmin[2] + self.intrinsic[0][2]
        pmin[1] = (pmin[1] * self.intrinsic[1][1]) / pmin[2] + self.intrinsic[1][2]
        pmin = torch.round(pmin).long()

        valid_ind_mask = torch.ge(pmax[0], 0) * torch.ge(pmax[1], 0) * \
                         torch.lt(pmax[0], self.image_dims[0]) * \
                         torch.lt(pmax[1], self.image_dims[1]) * \
                         torch.ge(pmin[0], 0) * torch.ge(pmin[1], 0) * \
                         torch.lt(pmin[0], self.image_dims[0]) * \
                         torch.lt(pmin[1], self.image_dims[1])

        if not valid_ind_mask.any():
            print('error: nothing projected in image space')
            return None

        pmax = pmax[:, valid_ind_mask]
        pmin = pmin[:, valid_ind_mask]
        lin_ind_volume = lin_ind_volume[valid_ind_mask]

        index_map = torch.empty((self.image_dims[1], self.image_dims[0]), dtype=torch.long).fill_(
            self.volume_dims[0] * self.volume_dims[1] * self.volume_dims[2])

        for i in range(lin_ind_volume.size(0)):
            index_map[pmin[1, i]:pmax[1, i], pmin[0, i]:pmax[0, i]] = torch.clamp(
                index_map[pmin[1, i]:pmax[1, i], pmin[0, i]:pmax[0, i]], max=lin_ind_volume[i])

        img_mask = torch.lt(index_map, self.volume_dims[0] * self.volume_dims[1] * self.volume_dims[2])
        self.show_projection(img_mask)

        index_map = torch.flatten(index_map)
        return index_map

    def forward(self, inp_occ_grid, lin_index_map):
        inp_occ_grid = inp_occ_grid[0]  # removes the channel dimension
        flatten_occ = torch.flatten(inp_occ_grid, start_dim=0, end_dim=-1).to(self.device)

        proj_img = torch.empty((self.image_dims[1] * self.image_dims[0]), dtype=torch.uint8).fill_(0).to(self.device)
        for i in range(lin_index_map.size(0)):
            if torch.lt(lin_index_map[i], flatten_occ.size(0)):
                proj_img[i] = flatten_occ[lin_index_map[i]]

        proj_img = torch.reshape(proj_img, (self.image_dims[1], self.image_dims[0]))

        return proj_img


# Inherit from Function
class Projection(Function):

    @staticmethod
    def forward(ctx, inp_occ_grid, lin_index_map, image_dims):
        inp_occ_grid = inp_occ_grid[0]  # removes the channel dimension
        flatten_occ = torch.flatten(inp_occ_grid, start_dim=0, end_dim=-1).to(self.device)

        ctx.save_for_backward(lin_index_map, flatten_occ)

        proj_img = torch.empty((self.image_dims[1] * self.image_dims[0]), dtype=torch.uint8).fill_(0).to(self.device)
        for i in range(lin_index_map.size(0)):
            if torch.lt(lin_index_map[i], flatten_occ.size(0)):
                proj_img[i] = flatten_occ[lin_index_map[i]]

        proj_img = torch.reshape(proj_img, (self.image_dims[1], self.image_dims[0]))

        return proj_img

    @staticmethod
    def backward(ctx, grad_output):
        lin_index_map, occ_grid = ctx.saved_variables

        for i in range(lin_index_map.size(0)):
            if torch.lt(lin_index_map[i], occ_grid.size(0)):
                occ_grid[lin_index_map[i]] = grad_output[i]

        return occ_grid


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt_file = "/media/sda2/shapenet/shapenet-voxelized-gt/02747177/fd013bea1e1ffb27c31c70b1ddc95e3f__0__.txt"
    gt_img_file = "/media/sda2/shapenet/renderings/02747177/fd013bea1e1ffb27c31c70b1ddc95e3f/color/color1.png"
    cam_file = "/media/sda2/shapenet/renderings/02747177/fd013bea1e1ffb27c31c70b1ddc95e3f/pose/pose1.json"

    gt_occ = loader.load_sample(gt_file).to(device)  # occupancy grid as pytorch tensor (shape: [1,D,H,W])
    gt_img = loader.load_img(gt_img_file).to(device)
    camera_to_world = loader.get_cam_to_world(cam_file).to(device)

    intrinsic = make_intrinsic().to(device)
    voxel_size = 1 / 32
    world_to_grid = make_world_to_grid().to(device)

    projection_helper = ProjectionHelper(device, intrinsic, 0.5, 1.5, [512, 512], [32, 32, 32], voxel_size)
    lin_index_map = projection_helper.compute_projection(gt_occ, camera_to_world, world_to_grid)
    proj_img = projection_helper.forward(gt_occ, lin_index_map)

    proj_loss = losses.vol_proj_loss(proj_img.unsqueeze(0).float(),
                                     gt_img.unsqueeze(0).float(), 1, device)
    print(proj_loss)

    img_mask = torch.eq(proj_img, 1)
    projection_helper.show_projection(img_mask)

    gt_img = gt_img.cpu().numpy().astype(np.uint8)
    gt_img[gt_img == 0] = 255  # 0 refers to not occupied -> so background color (white)
    gt_img[gt_img == 1] = 0  # 1 refers to occupied -> so silhouette color (black)

    img = Image.fromarray(gt_img, "L")
    img.show()
