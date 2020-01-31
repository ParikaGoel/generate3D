"""
Perspective Transformer Layer Implementation.

Transform the volume based on 4 x 4 perspective projection matrix.

Reference:
(1) "Perspective Transformer Nets: Perspective Transformer Nets:
Learning Single-View 3D Object Reconstruction without 3D Supervision."
Xinchen Yan, Jimei Yang, Ersin Yumer, Yijie Guo, Honglak Lee. In NIPS 2016
https://papers.nips.cc/paper/6206-perspective-transformer-nets-learning-single-view-3d-object-reconstruction-without-3d-supervision.pdf

(2) Official implementation in Torch: https://github.com/xcyan/ptnbhwd

(3) Reference implementation in TF:
https://github.com/tensorflow/models/tree/master/research/ptn

"""
import sys

sys.path.append('../.')
import config
import torch
import numpy as np
import dataset_loader as loader


def repeat(x, n_repeats):
    rep = np.transpose(
        np.expand_dims(np.ones(shape=np.stack([
            n_repeats,
        ])), 1), [1, 0])
    rep = rep.astype(int)
    x = np.matmul(np.reshape(x, (-1, 1)), rep)
    return np.reshape(x, [-1])


def interpolate(im, x, y, z, out_size):
    """Bilinear interploation layer.

Args:
  im: A 5D tensor of size [num_batch, depth, height, width, num_channels].
    It is the input volume for the transformation layer (tf.float32).
  x: A tensor of size [num_batch, out_depth, out_height, out_width]
    representing the inverse coordinate mapping for x (tf.float32).
  y: A tensor of size [num_batch, out_depth, out_height, out_width]
    representing the inverse coordinate mapping for y (tf.float32).
  z: A tensor of size [num_batch, out_depth, out_height, out_width]
    representing the inverse coordinate mapping for z (tf.float32).
  out_size: A tuple representing the output size of transformation layer
    (float).

Returns:
  A transformed tensor (tf.float32).

"""
    num_batch = im.shape[0]
    depth = im.shape[1]
    height = im.shape[2]
    width = im.shape[3]
    channels = im.shape[4]

    # Number of disparity interpolated.
    out_depth = out_size[0]
    out_height = out_size[1]
    out_width = out_size[2]
    zero = np.zeros([], dtype='int32')
    # 0 <= z < depth, 0 <= y < height & 0 <= x < width.
    max_z = im.shape[1] - 1
    max_y = im.shape[2] - 1
    max_x = im.shape[3] - 1

    # Converts scale indices from [-1, 1] to [0, width/height/depth].
    x = (x + 1.0) * (width) / 2.0
    y = (y + 1.0) * (height) / 2.0
    z = (z + 1.0) * (depth) / 2.0

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    z0 = np.floor(z).astype(int)
    z1 = z0 + 1

    x0_clip = np.clip(x0, zero, max_x)
    x1_clip = np.clip(x1, zero, max_x)
    y0_clip = np.clip(y0, zero, max_y)
    y1_clip = np.clip(y1, zero, max_y)
    z0_clip = np.clip(z0, zero, max_z)
    z1_clip = np.clip(z1, zero, max_z)
    dim3 = width
    dim2 = width * height
    dim1 = width * height * depth
    base = repeat(np.arange(num_batch) * dim1, out_depth * out_height * out_width)
    base_z0_y0 = base + z0_clip * dim2 + y0_clip * dim3
    base_z0_y1 = base + z0_clip * dim2 + y1_clip * dim3
    base_z1_y0 = base + z1_clip * dim2 + y0_clip * dim3
    base_z1_y1 = base + z1_clip * dim2 + y1_clip * dim3

    idx_z0_y0_x0 = base_z0_y0 + x0_clip
    idx_z0_y0_x1 = base_z0_y0 + x1_clip
    idx_z0_y1_x0 = base_z0_y1 + x0_clip
    idx_z0_y1_x1 = base_z0_y1 + x1_clip
    idx_z1_y0_x0 = base_z1_y0 + x0_clip
    idx_z1_y0_x1 = base_z1_y0 + x1_clip
    idx_z1_y1_x0 = base_z1_y1 + x0_clip
    idx_z1_y1_x1 = base_z1_y1 + x1_clip

    # Use indices to lookup pixels in the flat image and restore
    # channels dim
    im_flat = np.reshape(im, np.stack([-1, channels]))
    i_z0_y0_x0 = np.take(im_flat, idx_z0_y0_x0)
    i_z0_y0_x1 = np.take(im_flat, idx_z0_y0_x1)
    i_z0_y1_x0 = np.take(im_flat, idx_z0_y1_x0)
    i_z0_y1_x1 = np.take(im_flat, idx_z0_y1_x1)
    i_z1_y0_x0 = np.take(im_flat, idx_z1_y0_x0)
    i_z1_y0_x1 = np.take(im_flat, idx_z1_y0_x1)
    i_z1_y1_x0 = np.take(im_flat, idx_z1_y1_x0)
    i_z1_y1_x1 = np.take(im_flat, idx_z1_y1_x1)

    # Finally calculate interpolated values.
    x0_f = x0.astype(float)
    x1_f = x1.astype(float)
    y0_f = y0.astype(float)
    y1_f = y1.astype(float)
    z0_f = z0.astype(float)
    z1_f = z1.astype(float)
    # Check the out-of-boundary case.
    x0_valid = (np.less_equal(x0, max_x) & np.greater_equal(x0, 0)).astype(float)
    x1_valid = (np.less_equal(x1, max_x) & np.greater_equal(x1, 0)).astype(float)
    y0_valid = (np.less_equal(y0, max_y) & np.greater_equal(y0, 0)).astype(float)
    y1_valid = (np.less_equal(y1, max_y) & np.greater_equal(y1, 0)).astype(float)
    z0_valid = (np.less_equal(z0, max_z) & np.greater_equal(z0, 0)).astype(float)
    z1_valid = (np.less_equal(z1, max_z) & np.greater_equal(z1, 0)).astype(float)

    w_z0_y0_x0 = np.expand_dims(((x1_f - x) * (y1_f - y) *
                                 (z1_f - z) * x1_valid * y1_valid * z1_valid),
                                1)
    w_z0_y0_x1 = np.expand_dims(((x - x0_f) * (y1_f - y) *
                                 (z1_f - z) * x0_valid * y1_valid * z1_valid),
                                1)
    w_z0_y1_x0 = np.expand_dims(((x1_f - x) * (y - y0_f) *
                                 (z1_f - z) * x1_valid * y0_valid * z1_valid),
                                1)
    w_z0_y1_x1 = np.expand_dims(((x - x0_f) * (y - y0_f) *
                                 (z1_f - z) * x0_valid * y0_valid * z1_valid),
                                1)
    w_z1_y0_x0 = np.expand_dims(((x1_f - x) * (y1_f - y) *
                                 (z - z0_f) * x1_valid * y1_valid * z0_valid),
                                1)
    w_z1_y0_x1 = np.expand_dims(((x - x0_f) * (y1_f - y) *
                                 (z - z0_f) * x0_valid * y1_valid * z0_valid),
                                1)
    w_z1_y1_x0 = np.expand_dims(((x1_f - x) * (y - y0_f) *
                                 (z - z0_f) * x1_valid * y0_valid * z0_valid),
                                1)
    w_z1_y1_x1 = np.expand_dims(((x - x0_f) * (y - y0_f) *
                                 (z - z0_f) * x0_valid * y0_valid * z0_valid),
                                1)

    output = np.multiply(np.transpose(w_z0_y0_x0),i_z0_y0_x0) + np.multiply(np.transpose(w_z0_y0_x1), i_z0_y0_x1) + np.multiply(np.transpose(w_z0_y1_x0),i_z0_y1_x0) + np.multiply(np.transpose(w_z0_y1_x1), i_z0_y1_x1) + np.multiply(np.transpose(w_z1_y0_x0),i_z1_y0_x0) + np.multiply(np.transpose(w_z1_y0_x1), i_z1_y0_x1) + np.multiply(np.transpose(w_z1_y1_x0), i_z1_y1_x0) + np.multiply(np.transpose(w_z1_y1_x1), i_z1_y1_x1)

    return output


def create_grid(depth, height, width, z_near, z_far):
    x_t = np.reshape(
        np.tile(np.linspace(-1.0, 1.0, width), [height * depth]),
        [depth, height, width])
    y_t = np.reshape(
        np.tile(np.linspace(-1.0, 1.0, height), [width * depth]),
        [depth, width, height])
    y_t = np.transpose(y_t, [0, 2, 1])
    sample_grid = np.tile(
        np.linspace(float(z_near), float(z_far), depth), [width * height])
    z_t = np.reshape(sample_grid, [height, width, depth])
    z_t = np.transpose(z_t, [2, 0, 1])

    z_t = 1 / z_t
    d_t = 1 / z_t
    x_t /= z_t
    y_t /= z_t

    x_t_flat = np.reshape(x_t, (1, -1))
    y_t_flat = np.reshape(y_t, (1, -1))
    d_t_flat = np.reshape(d_t, (1, -1))

    ones = np.ones_like(x_t_flat)
    grid = np.concatenate((d_t_flat, y_t_flat, x_t_flat, ones), 0)
    return grid


def transform(theta, input, out_size, z_near, z_far):
    num_batch = input.shape[0]
    num_channels = input.shape[4]
    theta = np.reshape(theta, (-1, 4, 4))

    out_depth = out_size[0]
    out_height = out_size[1]
    out_width = out_size[2]
    grid = create_grid(out_depth, out_height, out_width, z_near, z_far)
    grid = np.expand_dims(grid, 0)
    grid = np.reshape(grid, [-1])
    grid = np.tile(grid, np.stack([num_batch]))
    grid = np.reshape(grid, np.stack([num_batch, 4, -1]))

    # Transform A x (x_t', y_t', 1, d_t)^T -> (x_s, y_s, z_s, 1).
    t_g = np.matmul(theta, grid)
    z_s = t_g[:, 0, :]
    y_s = t_g[:, 1, :]
    x_s = t_g[:, 2, :]

    z_s_flat = np.reshape(z_s, [-1])
    y_s_flat = np.reshape(y_s, [-1])
    x_s_flat = np.reshape(x_s, [-1])

    input_transformed = interpolate(input, x_s_flat, y_s_flat, z_s_flat,
                                    out_size)

    output = np.reshape(
        input_transformed,
        np.stack([num_batch, out_depth, out_height, out_width, num_channels]))

    return output


def transformer(voxels,
                theta,
                out_size,
                z_near,
                z_far):
    """
  Perspective Transformer Layer.

  :param voxels: A tensor of size [batch, y, z, x, channel].
  :param  theta: A tensor of size [num_batch, 16].
      It is the inverse camera transformation matrix (4 x 4) flattened into (1 x 16) .
  :param  out_size: A tuple representing the size of output of
      transformer layer.
  :param  z_near: A number representing the near clipping plane.
  :param  z_far: A number representing the far clipping plane.

  :returns
    A transformed tensor

  """

    output = transform(theta, voxels, out_size, z_near, z_far)
    return output


if __name__ == '__main__':
    z_near = 0.866
    z_far = 1.732
    gt_voxel = "../../../Assets_remote/shapenet-voxelized-gt/02747177/fd013bea1e1ffb27c31c70b1ddc95e3f__0__.txt"
    gt_occ = loader.load_sample(gt_voxel)
    gt_occ = np.expand_dims(gt_occ.numpy(),0)
    # gt_occ = gt_occ.unsqueeze(0)
    gt_occ = np.transpose(gt_occ, [0, 3, 2, 4, 1])

    theta = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.2, 0.0, 0.0, 0.0, 1.0])
    theta = np.expand_dims(theta, 0)
    output = transformer(gt_occ, theta, [32, 32, 32], z_near, z_far)
    output = np.transpose(output, [0,4,2,1,3])
    output = output[0]

    output = torch.from_numpy(output)
    loader.save_sample("../../../Assets/test.ply",output)
