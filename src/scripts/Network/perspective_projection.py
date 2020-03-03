import sys

sys.path.append('../.')
import os
import torch
import config
import losses
import voxel_grid
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

    def compute_index_mapping(self, occ_grid, world_to_camera):
        index_map = torch.empty((self.image_dims[1], self.image_dims[0]), dtype=int).fill_(-1).to(self.device)
        occ = torch.flatten(occ_grid, start_dim=0, end_dim=-1)
        occ = torch.nn.Sigmoid()(occ)
        occ_mask = torch.gt(occ, 0.2)

        if not occ_mask.any():
            print("error: Occupancy grid is empty")
            return torch.flatten(index_map, start_dim=0, end_dim=-1)

        lin_ind_volume = torch.arange(0, self.volume_dims[0] * self.volume_dims[1] * self.volume_dims[2],
                                      out=torch.LongTensor()).to(self.device)
        lin_ind_volume = lin_ind_volume[occ_mask]
        grid_coords = self.grid_to_world.new(4, lin_ind_volume.size(0))
        grid_coords[2] = lin_ind_volume / (self.volume_dims[0] * self.volume_dims[1])
        tmp = lin_ind_volume - (grid_coords[2] * self.volume_dims[0] * self.volume_dims[1]).long()
        grid_coords[1] = tmp / self.volume_dims[0]
        grid_coords[0] = torch.remainder(tmp, self.volume_dims[0])
        grid_coords[3].fill_(1)

        # cube vertex coords
        vertex_coords = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1],
                                      [0, 0, 1, 1, 0, 0, 1, 1],
                                      [0, 0, 0, 0, 1, 1, 1, 1],
                                      [0, 0, 0, 0, 0, 0, 0, 0]]).to(self.device)
        vertex_coords = torch.transpose(vertex_coords, 0, 1)

        corners = torch.stack([grid_coords.int() + vertex_coords[i][:, None] for i in range(8)])

        # transform to current frame
        camera_coords = torch.matmul(world_to_camera, torch.matmul(self.grid_to_world, corners.float()))
        depth = camera_coords.new(camera_coords.size(2)+1)
        depth[:-1] = torch.min(torch.abs(camera_coords[:, 2, :]), dim=0).values
        depth[-1] = torch.max(depth[:-1]) + 1

        # project into image
        p = camera_coords.clone()
        p[:, 1, :] = -p[:, 1, :]  # perspective projection flips the image in y- direction
        p[:, 0] = (p[:, 0] * self.intrinsic[0][0]) / torch.abs(p[:, 2]) + self.intrinsic[0][2]
        p[:, 1] = (p[:, 1] * self.intrinsic[1][1]) / torch.abs(p[:, 2]) + self.intrinsic[1][2]
        p = torch.round(p).long()[:, 0:2, :]

        p = torch.clamp(p, min=0, max=511)
        pmin = torch.min(p, dim=0).values
        pmax = torch.max(p, dim=0).values

        index_map = index_map.fill_(lin_ind_volume.size(0))

        for i in range(lin_ind_volume.size(0)):
            vals = index_map[pmin[1, i]:pmax[1, i], pmin[0, i]:pmax[0, i]]
            vals = torch.flatten(vals)
            depth_vals = torch.index_select(depth, 0, vals)
            mask = torch.gt(depth_vals, depth[i])
            vals[mask] = i
            vals = vals.reshape((pmax[1, i]-pmin[1, i], pmax[0, i]-pmin[0, i]))
            index_map[pmin[1, i]:pmax[1, i], pmin[0, i]:pmax[0, i]] = vals

        lin_index_map = torch.flatten(index_map, start_dim=0, end_dim=-1)
        invalid_mask = torch.eq(lin_index_map, lin_ind_volume.size(0))
        lin_index_map[invalid_mask] = 0
        lin_index_map = torch.index_select(lin_ind_volume, 0, lin_index_map)
        lin_index_map[invalid_mask] = -1

        return lin_index_map

    def index_mapping_n_views(self, occ_grid, poses):
        lin_index_map = torch.stack([self.compute_index_mapping(occ_grid, pose) for pose in poses])
        return lin_index_map

    def index_mapping_batch_n_views(self, occ_batch, poses_batch):
        batch_size = poses_batch.size(0)

        batch_index_map = torch.stack([self.index_mapping_n_views(occ_batch[i], poses_batch[i])
                                       for i in range(batch_size)])

        return batch_index_map

    def compute_projection(self, occ_grid, index_map):
        invalid_mask = torch.eq(index_map, -1)
        index_map[invalid_mask] = 0

        occ = torch.flatten(occ_grid, start_dim=0, end_dim=-1).float()
        proj_img = torch.index_select(occ, 0, index_map)
        proj_img[invalid_mask] = 0
        index_map[invalid_mask] = -1
        # self.show_projection(proj_img)

        return proj_img

    def project_occ_n_views(self, occ_grid, index_maps):
        projs = torch.stack([self.compute_projection(occ_grid, index_map) for index_map in index_maps])
        return projs

    def project_batch_n_views(self, occ_batch, poses_batch):
        batch_size = occ_batch.size(0)
        batch_index_map = self.index_mapping_batch_n_views(occ_batch, poses_batch)

        batch_projs = torch.stack([self.project_occ_n_views(occ_batch[i], batch_index_map[i])
                                   for i in range(batch_size)])

        return batch_index_map, batch_projs

    def copy_grad_to_occ(self, index_map, grad):
        output = torch.zeros(self.volume_dims[0] * self.volume_dims[1] * self.volume_dims[2],
                             dtype=torch.float).to(self.device)

        valid_mask = torch.gt(index_map, -1)
        grad = grad[valid_mask].float()
        index_map = index_map[valid_mask]

        output[index_map] = grad
        return output

    def copy_grad_n_views_occ(self, index_maps, grads):
        n_views = grads.size(0)
        output = torch.stack([self.copy_grad_to_occ(index_maps[i], grads[i])
                              for i in range(n_views)])
        output = torch.mean(output, dim=0)
        return output

    def show_projection(self, proj_img):
        img_mask = torch.gt(proj_img, 0.5)

        proj_img = proj_img.fill_(255)
        proj_img[img_mask] = 0

        proj_img = torch.reshape(proj_img, (self.image_dims[1], self.image_dims[0]))

        img_np = proj_img.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img_np, 'L')
        img.show()

    def save_projection(self, file, proj_img):
        img = proj_img.clone()
        img = img.detach()
        img_mask = torch.gt(img, 0.5)
        img = img.fill_(255)
        img[img_mask] = 0

        img = torch.reshape(img, (self.image_dims[1], self.image_dims[0]))

        img_np = img.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img_np, 'L')
        img.save(file)

    def save_gradient(self, file, grad):
        mask_pos = torch.gt(grad, 1e-10)
        mask_neg = torch.lt(grad, -1e-10)
        grad_vis = torch.empty((512 * 512), dtype=torch.uint8).fill_(1)
        grad_vis[mask_pos] = 0
        grad_vis[mask_neg] = 150
        grad_vis = torch.reshape(grad_vis, (512, 512))

        grad_vis = grad_vis.cpu().numpy().astype(np.uint8)
        grad_vis[grad_vis == 1] = 255
        vis = Image.fromarray(grad_vis, 'L')
        vis.save(file)

    def save_gradient_n_views(self, folder, grads):
        n_views = grads.size(0)

        for v in range(n_views):
            file = os.path.join(folder, "img_grad%02d.png" % v)
            self.save_gradient(file, grads[v])

    def save_gradient_occ(self, folder, grad_occ):
        file = os.path.join(folder, "occ_grad.txt")
        grads = grad_occ[0].cpu().numpy().transpose(2, 1, 0)
        positions = np.where(grads > 1e-10)
        pos_neg = np.where(grads < -1e-10)
        with open(file, "w") as f:
            for i, j, k in zip(*positions):
                data = np.column_stack((i, j, k, 0, 0, 0))
                np.savetxt(f, data, fmt='%d %d %d %d %d %d', delimiter=' ')
            for i, j, k in zip(*pos_neg):
                data1 = np.column_stack((i, j, k, 150, 150, 150))
                np.savetxt(f, data1, fmt='%d %d %d %d %d %d', delimiter=' ')

        ply_file = os.path.join(folder, "occ_grad.ply")
        voxel_grid.txt_to_mesh(file, ply_file)


# Inherit from Function
class Projection(Function):

    @staticmethod
    def visualize(grads, img_grad=True):
        folder = "/home/parika/WorkingDir/complete3D/Assets/output-network/data/model7/grads"
        projection_helper = ProjectionHelper()

        if img_grad:
            projection_helper.save_gradient_n_views(folder, grads[0])
        else:
            projection_helper.save_gradient_occ(folder, grads[0])

    @staticmethod
    def forward(ctx, occ_grid, poses):
        projection_helper = ProjectionHelper()
        batch_index_map, batch_projs = projection_helper.project_batch_n_views(occ_grid, poses)

        ctx.save_for_backward(batch_index_map)
        return batch_projs

    @staticmethod
    def backward(ctx, grads):
        # saving the gradient
        Projection.visualize(grads)
        batch_index_map = ctx.saved_tensors
        batch_index_map = batch_index_map[0]
        batch_size = grads.size(0)
        projection_helper = ProjectionHelper()
        output = torch.stack([torch.reshape(projection_helper.copy_grad_n_views_occ(batch_index_map[b], grads[b]),
                                            (1, 32, 32, 32)) for b in range(batch_size)])

        Projection.visualize(output, False)

        return output, None

# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     rainbow_file = "../../../Assets_remote/test/rainbow_frustrum_.txt"
#     pose_folder = "/home/parika/WorkingDir/complete3D/Assets/shapenet-renderings/02747177/501154f25599ee80cb2a965e75be701c/pose/"
#     gt_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-voxelized-gt/02747177/501154f25599ee80cb2a965e75be701c__0__.txt"
#     img_folder = "/home/parika/WorkingDir/complete3D/Assets/shapenet-renderings/02747177/501154f25599ee80cb2a965e75be701c/color/"
#
#     proj_img_out = "/home/parika/WorkingDir/complete3D/Assets/output-network/data/proj_imgs"
#     gt_img_out = "/home/parika/WorkingDir/complete3D/Assets/output-network/data/gt_imgs"
#     out_folder = "/home/parika/WorkingDir/complete3D/Assets/output-network/data"
#
#     gt_occ = loader.load_sample(gt_file).to(device).unsqueeze(0)
#     poses = loader.load_poses(pose_folder).to(device).unsqueeze(0)
#     gt_imgs = loader.load_imgs(img_folder).to(device).unsqueeze(0)
#
#     # poses = poses[:, 10, :, :].unsqueeze(0)
#     # gt_imgs = gt_imgs[:, 10, :].unsqueeze(0)
#     projection_helper = ProjectionHelper()
#     index_maps, projs = projection_helper.project_batch_n_views(gt_occ, poses)
#
#     # grad = torch.from_numpy(np.loadtxt('grad.txt')).float().to(device).unsqueeze(0)
#     output_occ = projection_helper.copy_grad_n_views_occ(index_maps[0], gt_imgs[0])
#     projection_helper.save_gradient_occ(out_folder, output_occ)
#     #
#     #
#     for img_idx, proj_img in enumerate(projs[0]):
#         projection_helper.save_projection(os.path.join(proj_img_out, "img_%02d.png" % img_idx), proj_img, True)
#
#     for img_idx, img_gt in enumerate(gt_imgs[0]):
#         projection_helper.save_projection(os.path.join(gt_img_out, "img_%02d.png" % img_idx), img_gt, True)
#
#     loss = losses.proj_loss(projs, gt_imgs, 1, device)
#
#     grad = torch.from_numpy(np.loadtxt('grad_out.txt')).float().to(device).unsqueeze(0)
#     occ = projection_helper.backward(grad, gt_occ, index_maps)
