import json
import numpy as np


# Creates a camera objects, loads its parameters from json file and returns that object
def load_camera(json_file):
    cam = Camera()
    cam.load_from_json_file(json_file)
    return cam


# For now the center is always at the center of the image, so half of the resolution
class Camera:
    def __init__(self, resolution=[512, 512], focal=[525.0, 525.0], z_near=0.01, z_far=1000.0):
        self._focal = focal
        self._resolution = resolution
        self._center = [(res / 2) for res in resolution]
        self._znear = z_near
        self._zfar = z_far
        self._extrinsic = np.identity(4)  # extrinsic matrix that gives the transformation from world to camera system
        self._pose = np.identity(4) # camera pose : position of camera in world system
        self._intrinsic = np.identity(3)

    @property
    def focal(self):
        return self._focal

    @property
    def resolution(self):
        return self._resolution

    @property
    def center(self):
        return self._center

    @property
    def z_near(self):
        return self._znear

    @property
    def z_far(self):
        return self._zfar

    @property
    def intrinsic(self):
        return self._intrinsic

    @property
    def extrinsic(self):
        return self._extrinsic

    @property
    def pose(self):
        return self._pose

    def fov(self):
        fov = 2.0 * np.degrees(
            np.arctan((self._resolution / 2.0) / self._focal))
        return fov
    
    def load_from_json_file(self, filename):
        with open(filename) as camera_file:
            data = json.load(camera_file)

            # read the intrinsic parameters
            self._resolution[0] = data['intrinsic']['width']
            self._resolution[1] = data['intrinsic']['height']

            # intr = np.empty(0)
            # for val in data['intrinsic']['intrinsic_matrix']:
            #     intr = np.append(intr, val)
            # self._intrinsic = np.transpose(np.reshape(intr, (3, 3)))
            #
            # self._focal[0] = self._intrinsic[0][0]
            # self._focal[1] = self._intrinsic[1][1]
            # self._center[0] = self._intrinsic[0][2]
            # self._center[1] = self._intrinsic[1][2]

            self._focal[0] = data['intrinsic']['fx']
            self._focal[1] = data['intrinsic']['fy']
            self._znear = data['intrinsic']['z_near']
            self._zfar = data['intrinsic']['z_far']
            self._center = [(res / 2) for res in self._resolution]

            # set the intrinsic matrix
            self._intrinsic[0][0] = self._focal[0]
            self._intrinsic[1][1] = self._focal[1]
            self._intrinsic[0][2] = self._center[0]
            self._intrinsic[1][2] = self._center[1]

            # read the camera pose
            pose = np.empty(0)
            for val in data['pose']:
                pose = np.append(pose, val)
            self._pose = np.transpose(np.reshape(pose, (4, 4)))

            # extrinsic matrix will be the inverse of camera pose
            self._extrinsic = np.linalg.inv(self._pose)
