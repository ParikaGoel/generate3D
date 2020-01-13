import numpy as np
from PIL import Image

if __name__ == '__main__':
    depth_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-renderings/02747177/85d8a1ad55fa646878725384d6baf445/depth/depth0.png"
    color_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-renderings/02747177/85d8a1ad55fa646878725384d6baf445/color/color0.png"

    im = Image.open(color_file)
    np_img = np.array(im)

    depth_im = Image.open(depth_file)
    d_np_img = np.array(depth_im)
    print(d_np_img.shape)