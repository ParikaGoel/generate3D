import torch
from torch import nn


# improved upon Net2
class Net3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net3D, self).__init__()

        # Encoder Part
        self.conv1 = nn.Conv3d(in_channels, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv3d(16, 36, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv3d(36, 72, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv3d(72, 144, kernel_size=3, stride=1, padding=0)
        self.conv6 = nn.Conv3d(144, 240, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        # Decoder Part
        self.deconv1 = nn.ConvTranspose3d(240, 144, kernel_size=3, stride=1, padding=0, output_padding=0)
        self.deconv2 = nn.ConvTranspose3d(144, 72, kernel_size=3, stride=1, padding=0, output_padding=0)
        self.deconv3 = nn.ConvTranspose3d(72, 36, kernel_size=3, stride=1, padding=0, output_padding=0)
        self.deconv4 = nn.ConvTranspose3d(36, 16, kernel_size=3, stride=1, padding=0, output_padding=0)
        self.deconv5 = nn.ConvTranspose3d(16, 8, kernel_size=3, stride=1, padding=0, output_padding=0)
        self.deconv6 = nn.ConvTranspose3d(8, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # Encoder Part : [1, 32, 32, 32] -> [256, 6, 6, 6]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))

        # Decoder Part : [256, 6, 6, 6] -> [1, 32, 32, 32]
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.relu(self.deconv4(x))
        x = self.relu(self.deconv5(x))
        x = self.deconv6(x)

        return x


# using info from encoder in decoder something like unet
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        # Encoder Part
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Decoder Part
        self.deconv1 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose3d(128 * 2, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose3d(64 * 2, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose3d(32 * 2, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # Encoder Part : [1, 32, 32, 32] -> [256, 2, 2, 2]
        print(x.shape)
        enc1 = self.relu(self.conv1(x))
        print(enc1.shape)
        enc2 = self.relu(self.conv2(enc1))
        print(enc2.shape)
        enc3 = self.relu(self.conv3(enc2))
        print(enc3.shape)
        enc4 = self.relu(self.conv4(enc3))
        print(enc4.shape)

        # Decoder Part : [256, 2, 2, 2] -> [1, 32, 32, 32]
        dec1 = self.relu(self.deconv1(enc4))
        print(dec1.shape)
        dec1 = torch.cat((dec1,enc3), dim=1)
        print(dec1.shape)
        dec2 = self.relu(self.deconv2(dec1))
        print(dec2.shape)
        dec2 = torch.cat((dec2, enc2), dim=1)
        print(dec2.shape)
        dec3 = self.relu(self.deconv3(dec2))
        print(dec3.shape)
        dec3 = torch.cat((dec3, enc1), dim=1)
        print(dec3.shape)
        out = self.deconv4(dec3)
        print(out.shape)

        return out