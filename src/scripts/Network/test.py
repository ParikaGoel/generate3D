import torch
import numpy as np
from PIL import Image
import dataset_loader as loader


if __name__ == '__main__':
    color_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-renderings/04379243/fff7f07d1c4042f8a946c24c4f9fb58e/color/color00.png"
    img = torch.reshape(loader.load_img(color_file),(1,512,512)).float()
    print(img.shape)