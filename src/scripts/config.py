depth_factor = 100

# rendering images
img_width = 512
img_height = 512
znear = 0.4
zfar = 4.0
focal = 525.0
cam_depth = 1.4
color_bg = [1., 1., 1.]

# voxel
vox_dim = 32

# train network
num_epochs = 50
batch_size = 32
lr = 0.001
momentum = 0.9
weight_decay = 1e-5
trunc_dist = 4
n_vis = 10  # number of samples to visualize while training
