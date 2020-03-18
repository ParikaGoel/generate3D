import glob
import trimesh
import JSONHelper
import numpy as np
from mesh_to_sdf import mesh_to_voxels

params = JSONHelper.read("parameters.json")

def generate_sdf(synset_id, model_id):
    obj_file = params["shapenet"] + synset_id + "/" + model_id + "/models/model_normalized.obj"
    sdf_file = params["shapenet_voxelized"] + synset_id + "/" + model_id
    mesh = trimesh.load(obj_file)
    voxels = mesh_to_voxels(mesh, 32, pad=False)
    np.save(sdf_file, voxels)


if __name__ == '__main__':
    synset_lst = ["03001627"]
    failed_cases = {}
    file = params["shapenet_raytraced"] + "failed_cases.json"

    for synset_id in synset_lst:
        for f in glob.glob(params["shapenet"] + synset_id + "/*/models/model_normalized.obj"):
            model_id = f.split("/", 10)[8]
            print(synset_id, " : ", model_id)

            try:
                if not synset_id in failed_cases.keys():
                    failed_cases[synset_id] = []

                generate_sdf(synset_id, model_id)
            except:
                failed_cases[synset_id].append(model_id)
                pass

        print("Finished generating sdf synset ", synset_id)

    JSONHelper.write(file, failed_cases)
