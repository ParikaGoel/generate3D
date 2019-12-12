import torch
from torch import nn

class SingleConvLayer(nn.Sequential):
    """
    Module to create two consecutive convolutional layers with relu non linearity
    (Conv3d -> RelU)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(SingleConvLayer, self).__init__()

        self.add_module('conv1', nn.Conv3d(in_channels, out_channels, kernel_size))
        self.add_module('relu1', nn.ReLU(inplace=True))


class DoubleConvLayer(nn.Sequential):
    """
    Module to create two consecutive convolutional layers with relu non linearity
    (Conv3d -> RelU -> Conv3d -> RelU)
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, encoder=True):
        super(DoubleConvLayer, self).__init__()

        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels

            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we are in the decoder path, decrease the number of channels in the first convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels


        # conv1
        self.add_module('conv1', nn.Conv3d(conv1_in_channels, conv1_out_channels, kernel_size))
        self.add_module('relu1', nn.ReLU(inplace=True))

        # conv2
        self.add_module('conv2', nn.Conv3d(conv2_in_channels, conv2_out_channels, kernel_size))
        self.add_module('relu2', nn.ReLU(inplace=True))


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, pool_kernel_size=2):
        super(Encoder, self).__init__()
        self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
        self.basic_block = SingleConvLayer(in_channels, out_channels, conv_kernel_size)

    def forward(self,x):
        x = self.basic_block(x)
        x = self.pooling(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Decoder, self).__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.basic_block = SingleConvLayer(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.upsample(x)
        x = self.basic_block(x)