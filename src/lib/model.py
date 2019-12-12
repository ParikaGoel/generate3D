import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()

        # Encoder Part
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=8, stride=2, padding=3)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Decoder Part
        self.upsample = nn.ConvTranspose3d(128, 128, kernel_size=6, stride=2, padding=2)
        self.conv4 = nn.Conv3d(128, 64, kernel_size=5, stride=1, padding=2)
        self.conv5 = nn.Conv3d(64, 32, kernel_size=7, stride=1, padding=3)

        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder Part : [1, 32, 32, 32] -> [128, 4, 4, 4]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Decoder Part
        # First do upsampling to increase the width, height, depth to initial size : [128, 4, 4, 4] -> [128, 32, 32, 32]
        # convolutional layers to decrease the number of channels : [128, 32, 32, 32] -> [32, 32, 32, 32]
        x = self.upsample(self.upsample(self.upsample(x)))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        # Final Convolutional Layer which reduces the number of channels to initial number : [32, 32, 32, 32] -> [1,
        # 32, 32, 32]
        x = self.final_conv(x)
        return x

