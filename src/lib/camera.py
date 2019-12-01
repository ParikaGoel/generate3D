import json
import numpy as np


# Creates a camera objects, loads its parameters from json file and returns that object
def load_camera(json_file):
    cam = Camera()
    cam.load_from_json_file(json_file)
    return cam


class Camera:
    def __init__(self, fx=525.0, fy=525.0, cx=319.5, cy=239.5, width=640, height=480):
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self._width = width
        self._height = height
        self._extrinsic = np.identity(4)  # extrinsic matrix that gives the transformation from world to camera system
        self._intrinsic = np.identity(3)

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height

    def get_fx(self):
        return self._fx

    def get_fy(self):
        return self._fy

    def get_cx(self):
        return self._cx

    def get_cy(self):
        return self._cy

    def get_intrinsic_matrix(self):
        return self._intrinsic

    def get_extrinsic_matrix(self):
        return self._extrinsic
    
    def load_from_json_file(self, filename):
        with open(filename) as camera_file:
            data = json.load(camera_file)

            # read the intrinsic parameters
            self._width = data['intrinsic']['width']
            self._height = data['intrinsic']['height']

            intr = np.empty(0)
            for val in data['intrinsic']['intrinsic_matrix']:
                intr = np.append(intr, val)
            self._intrinsic = np.transpose(np.reshape(intr, (3, 3)))

            self._fx = self._intrinsic[0][0]
            self._fy = self._intrinsic[1][1]
            self._cx = self._intrinsic[0][2]
            self._cy = self._intrinsic[1][2]

            # read the extrinsic parameters
            extr = np.empty(0)
            for val in data['extrinsic']:
                extr = np.append(extr, val)
            self._extrinsic = np.transpose(np.reshape(extr, (4, 4)))
