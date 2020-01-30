import sys
sys.path.append("../.")
import config


"""3D->2D projector model as used in PTN (NIPS16)."""

def project(voxels, transform_matrix):
  """
  Model transforming the 3D voxels into 2D projections.

  :param voxels: A tensor of size [batch, channel, depth, height, width]
      representing the input of projection layer.
  :param transform_matrix: A tensor of size [batch, 16] representing
      the flattened 4-by-4 matrix for transformation.

  :return
    A transformed tensor

  """
  # ToDo(Parika): Rearrangement done in original code, check how to apply it for your implementation
  # Rearrangement (batch, z, y, x, channel) --> (batch, y, z, x, channel).
  # By the standard, projection happens along z-axis but the voxels
  # are stored in a different way. So we need to switch the y and z
  # axis for transformation operation.
  # voxels = tf.transpose(voxels, [0, 2, 1, 3, 4])
  # z_near = params.focal_length
  # z_far = params.focal_length + params.focal_range
  
  # my implementation voxel -> (batch, channel, z, y, x)
  # ref tf implementation requires (batch, y, z, x, channel)
  voxels = np.transpose(voxels, [0, 3, 2, 4, 1])   # need to change this after adapting projection implementation to my code
  z_near = config.znear
  z_far = config.zfar
  transformed_voxels = perspective_transform.transformer(
      voxels, transform_matrix, [params.vox_size] * 3, z_near, z_far)
  views = tf.reduce_max(transformed_voxels, [1])
  views = tf.reverse(views, [1])
  return views