import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from model import *


# function to show an image
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()


def main():
    net = Net(1,1)
    input = torch.randn(1,1,32,32,32)
    out = net(input)
    # print(out)


if __name__ == '__main__':
    main()