import torch
from torch import nn


class Net2d3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DFNet, self).__init__()

        # Encoder Part
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)  # try stride of 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(int(256 * 8 * 8), 8192)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU(inplace=True)

        # Decoder Part
        self.deconv1 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=3, padding=2)
        self.deconv2 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose3d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # Encoder Part : [1, 512, 512] -> [128, 4, 4, 4]
        n_batch = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.flatten(x)
        x = self.relu(self.fc(x))

        x = torch.reshape(x, (n_batch, 128, 4, 4, 4))

        # Decoder Part : [128, 4, 4, 4] -> [1, 32, 32, 32]
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.deconv3(x)

        return x


class Net2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net2, self).__init__()

        # Encoder Part
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1)  # try stride of 2
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Decoder Part
        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose3d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # Encoder Part : [1, 32, 32, 32] -> [256, 2, 2, 2]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        # Decoder Part : [256, 2, 2, 2] -> [1, 32, 32, 32]
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.deconv4(x)

        return x


# architecture taken from shapeComplete
class Net3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net3, self).__init__()

        # Encoder Part
        # try leaky relu instead of relu
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=4, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        # Decoder Part
        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=1, padding=0, output_padding=0)
        self.deconv2 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv3 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.deconv4 = nn.ConvTranspose3d(32, out_channels, kernel_size=4, stride=2, padding=1, output_padding=0)

    def forward(self, x):
        # Encoder Part : [1, 32, 32, 32] -> [256, 1, 1, 1]
        print(x.shape)
        x = self.relu(self.conv1(x))
        print(x.shape)
        x = self.relu(self.conv2(x))
        print(x.shape)
        x = self.relu(self.conv3(x))
        print(x.shape)
        x = self.relu(self.conv4(x))
        print(x.shape)

        # Decoder Part : [256, 1, 1, 1] -> [1, 32, 32, 32]
        x = self.relu(self.deconv1(x))
        print(x.shape)
        x = self.relu(self.deconv2(x))
        print(x.shape)
        x = self.relu(self.deconv3(x))
        print(x.shape)
        x = self.deconv4(x)
        print(x.shape)

        return x


class Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()

        # Encoder Part
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1)  # try stride of 2
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Decoder Part
        self.conv4 = nn.ConvTranspose3d(128, 64, kernel_size=3, stride=3, padding=2)
        self.conv5 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.ConvTranspose3d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # Encoder Part : [1, 32, 32, 32] -> [128, 4, 4, 4]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Decoder Part : [128, 4, 4, 4] -> [1, 32, 32, 32]
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)

        return x


class Net2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net2D, self).__init__()

        # Encoder Part
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)  # try stride of 2
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
