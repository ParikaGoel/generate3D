import sys
sys.path.append("../.")
import config
import renderer

if __name__ == '__main__':
    obj_file = "/home/parika/WorkingDir/complete3D/Assets/shapenet-data/02747177/fd013bea1e1ffb27c31c70b1ddc95e3f/models/model_normalized.obj"
    renderer.generate_images(obj_file, out, 10)
