import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()

        # Encoder Part
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)

        # Decoder Part
        self.conv4 = nn.ConvTranspose3d(128, 64, kernel_size=3, padding=1)
        self.conv5 = nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1)
        self.conv6 = nn.ConvTranspose3d(32, out_channels, kernel_size=3, padding=1)
        self.unpool = nn.MaxUnpool3d(2, stride=2)

        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder Part : [3, 32, 32, 32] -> [128, 4, 4, 4]
        x, indices1 = self.pool(self.relu(self.conv1(x)))
        x, indices2 = self.pool(self.relu(self.conv2(x)))
        x, indices3 = self.pool(self.relu(self.conv3(x)))

        # Decoder Part : [128, 4, 4, 4] -> [3, 32, 32, 32]
        x = self.unpool(x, indices3)
        x = self.relu(self.conv4(x))
        x = self.unpool(x, indices2)
        x = self.relu(self.conv5(x))
        x = self.unpool(x, indices1)
        x = self.conv6(x)

        return x

