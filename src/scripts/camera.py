import json
import numpy as np


# Creates a camera objects, loads its parameters from json file and returns that object
def load_camera(json_file):
    cam = Camera()
    cam.load_from_json_file(json_file)
    return cam


# For now the center is always at the center of the image, so half of the resolution
class Camera:
    def __init__(self, center=[256, 256], focal=[525.0, 525.0], z_near=0.01, z_far=1000.0):
        self._focal = focal
        self._resolution = [(c * 2) for c in center]
        self._center = center
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
            self._center[0] = data['intrinsic']['cx']
            self._center[1] = data['intrinsic']['cy']

            self._focal[0] = data['intrinsic']['fx']
            self._focal[1] = data['intrinsic']['fy']
            self._znear = data['intrinsic']['z_near']
            self._zfar = data['intrinsic']['z_far']
            self._resolution = [(c * 2) for c in self._center]

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

    def export_to_json(self, file_name):
        pose = self._pose.flatten(order='F')
        cam_data = \
            {
                'intrinsic':
                    {
                        'cx': int(self._center[0]),
                        'cy': int(self._center[1]),
                        'fx': float(self._focal[0]),
                        'fy': float(self._focal[1]),
                        'z_near': float(self._znear),
                        'z_far': float(self._zfar)
                    },
                'pose': pose.tolist()
            }

        with open(file_name, 'w') as fp:
            json.dump(cam_data, fp)
