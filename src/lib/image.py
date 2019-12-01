import numpy as np
import torchvision.transforms as transforms
import cv2


# takes numpy array image
def preprocess_image(img):
    # if the image has an alpha channel, remove it
    img_rgb = np.array(img)[:, :, :3].astype(np.float32)

    # scale image so that the values lie in the range (0,1)
    img_rgb = img_rgb / 255

    # Note : For now we dont need image as pytorch tensor, so keeping it as numpy array
    # # create a transform that will convert numpy image to torch tensor
    #
    # transform = transforms.Compose(
    #     [transforms.ToTensor()])
    #
    # # returns the transformed torch tensor of shape (channel x height x width)
    # return transform(img_rgb)
    return img_rgb


# load single image from disk
# returns numpy array image
def load_image(filename):
    img = cv2.imread(filename)
    img = preprocess_image(img)
    return img
