import os
import torch
import torchvision
import numpy as np
from model import *
from torch import nn
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.datasets import CIFAR10


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 32, 32)
    return x


# Writing our model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 6, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 3, kernel_size=5),
            nn.ReLU(True))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    # Loading and Transforming data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))])
    # trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    # dataloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=4)
    # testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    # defining some params
    num_epochs = 21

    # model = Net2D(3, 3).cpu()
    model = Autoencoder().cpu()
    distance = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

    # for epoch in range(num_epochs):
    #     for data in dataloader:
    #         img, _ = data
    #         img = Variable(img).cpu()
    #         # ===================forward=====================
    #         output = model(img)
    #         loss = distance(output, img)
    #         # ===================backward====================
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     # ===================log========================
    #     print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))
    #     # if epoch % 5 == 0:
    #     pic = to_img(output)
    #     save_image(pic, 'results/dc_img/image_{}.png'.format(epoch))
    #
    # torch.save(model.state_dict(), 'results/conv_autoencoder.pth')
    image_file = './results/dc_img/stool32.png'
    img = Image.open(image_file)
    img = transform(img)
    img = img.unsqueeze(0)
    img = Variable(img).cpu()

    for epoch in range(num_epochs):
        # ===================forward=====================
        output = model(img)
        loss = distance(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        if epoch % 10 == 0:
            pic = to_img(output)
            save_image(pic, 'results/dc_img/image_{}.png'.format(epoch))
