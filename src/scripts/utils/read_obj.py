import numpy as np
import tinyobjloader


if __name__ == '__main__':
    obj_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-data/04379243/1a00aa6b75362cc5b324368d54a7416f/models/model_normalized.obj"
    mtl_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-data/04379243/1a00aa6b75362cc5b324368d54a7416f/models/model_normalized.mtl"

    reader = tinyobjloader.ObjReader()
    ret = reader.ParseFromFile(obj_file)

    if ret == False:
        print("Warn:", reader.Warning())
        print("Err:", reader.Error())
        print("Failed to load : ", filename)

        sys.exit(-1)