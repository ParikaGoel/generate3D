import torch
from torch import nn


class Net3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net3D, self).__init__()

        # Encoder Part
        self.encode1 = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(8),
            nn.ReLU(True)
        )
        self.encode2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm3d(16),
            nn.ReLU(True)
        )
        self.encode3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm3d(32),
            nn.ReLU(True)
        )
        self.encode4 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm3d(64),
            nn.ReLU(True)
        )
        self.encode5 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm3d(128),
            nn.ReLU(True)
        )
        self.encode6 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm3d(256),
            nn.ReLU(True)
        )

        # Decoder Part
        self.decode1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.InstanceNorm3d(128),
            nn.ReLU(True)
        )
        self.decode2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.InstanceNorm3d(64),
            nn.ReLU(True)
        )
        self.decode3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.InstanceNorm3d(32),
            nn.ReLU(True)
        )
        self.decode4 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.InstanceNorm3d(16),
            nn.ReLU(True)
        )
        self.decode5 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, stride=1, padding=0, output_padding=0),
            nn.InstanceNorm3d(8),
            nn.ReLU(True)
        )
        self.decode6 = nn.ConvTranspose3d(8, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # Encoder Part : [1, 32, 32, 32] -> [256, 6, 6, 6]
        x = self.encode1(x)
        x = self.encode2(x)
        x = self.encode3(x)
        x = self.encode4(x)
        x = self.encode5(x)
        x = self.encode6(x)

        # Decoder Part : [256, 6, 6, 6] -> [1, 32, 32, 32]
        x = self.decode1(x)
        x = self.decode2(x)
        x = self.decode3(x)
        x = self.decode4(x)
        x = self.decode5(x)
        x = self.decode6(x)

        return x


# using info from encoder in decoder something like unet
# class UNet3D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(UNet3D, self).__init__()
#
#         # Encoder Part
#         self.encode1 = nn.Sequential(
#             nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
#             nn.InstanceNorm3d(32),
#             nn.ReLU(True)
#         )
#         self.encode2 = nn.Sequential(
#             nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.InstanceNorm3d(64),
#             nn.ReLU(True)
#         )
#         self.encode3 = nn.Sequential(
#             nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.InstanceNorm3d(128),
#             nn.ReLU(True)
#         )
#         self.encode4 = nn.Sequential(
#             nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.InstanceNorm3d(256),
#             nn.ReLU(True)
#         )
#
#         # Decoder Part
#         self.decode1 = nn.Sequential(
#             nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.InstanceNorm3d(128),
#             nn.ReLU(True)
#         )
#         self.decode2 = nn.Sequential(
#             nn.ConvTranspose3d(128 * 2, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.InstanceNorm3d(64),
#             nn.ReLU(True)
#         )
#         self.decode3 = nn.Sequential(
#             nn.ConvTranspose3d(64 * 2, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
#             nn.InstanceNorm3d(32),
#             nn.ReLU(True)
#         )
#         self.decode4 = nn.ConvTranspose3d(32 * 2, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
#
#     def forward(self, x):
#         # Encoder Part : [1, 32, 32, 32] -> [256, 2, 2, 2]
#         enc1 = self.encode1(x)
#         enc2 = self.encode2(enc1)
#         enc3 = self.encode3(enc2)
#         enc4 = self.encode4(enc3)
#
#         # Decoder Part : [256, 2, 2, 2] -> [1, 32, 32, 32]
#         dec1 = self.decode1(enc4)
#         dec1 = torch.cat((dec1,enc3), dim=1)
#         dec2 = self.decode2(dec1)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec3 = self.decode3(dec2)
#         dec3 = torch.cat((dec3, enc1), dim=1)
#         out = self.decode4(dec3)
#
#         return out

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        # Encoder Part
        self.encode1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(True)
        )
        self.encode2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(True)
        )
        self.encode3 = nn.Sequential(
            nn.Conv3d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(96),
            nn.ReLU(True)
        )
        self.encode4 = nn.Sequential(
            nn.Conv3d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(True)
        )

        # Decoder Part
        self.decode1 = nn.Sequential(
            nn.ConvTranspose3d(128, 96, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm3d(96),
            nn.ReLU(True)
        )
        self.decode2 = nn.Sequential(
            nn.ConvTranspose3d(96 * 2, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(True)
        )
        self.decode3 = nn.Sequential(
            nn.ConvTranspose3d(64 * 2, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(True)
        )
        self.decode4 = nn.ConvTranspose3d(32 * 2, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # Encoder Part : [1, 32, 32, 32] -> [256, 2, 2, 2]
        enc1 = self.encode1(x)
        enc2 = self.encode2(enc1)
        enc3 = self.encode3(enc2)
        enc4 = self.encode4(enc3)

        # Decoder Part : [256, 2, 2, 2] -> [1, 32, 32, 32]
        dec1 = self.decode1(enc4)
        dec1 = torch.cat((dec1,enc3), dim=1)
        dec2 = self.decode2(dec1)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec3 = self.decode3(dec2)
        dec3 = torch.cat((dec3, enc1), dim=1)
        out = self.decode4(dec3)

        return out