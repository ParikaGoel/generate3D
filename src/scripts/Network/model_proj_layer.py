import torch
from torch import nn
from perspective_projection import Projection


class Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()

        # Encoder Part
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1) # try stride of 2
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Decoder Part
        self.conv4 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=3, padding=2)
        self.conv5 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.ConvTranspose3d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)


    def forward(self, x, index_map):
        # Encoder Part : [1, 32, 32, 32] -> [128, 4, 4, 4]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Decoder Part : [128, 4, 4, 4] -> [1, 32, 32, 32]
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))

        # index_map : batch_size x num_views x (img_height * img_width)
        proj_imgs = Projection.apply(x, index_map)
        # this should give proj_imgs in the shape : batch_size x num_views x (img_height * img_width)

        return x, proj_imgs


class Net2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net2D, self).__init__()

        # Encoder Part
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1) # try stride of 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Decoder Part
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=3, padding=2)
        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)


    def forward(self, x):
        # Encoder Part : [3, 32, 32] -> [128, 4, 4]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Decoder Part : [128, 4, 4] -> [3, 32, 32]
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))

        return x

