import sys

assert sys.version_info >= (3, 5)

import os
import glob
import pathlib
import subprocess
import JSONHelper
import numpy as np
from config import *

if __name__ == '__main__':
    params = JSONHelper.read("./parameters.json")  # <-- read parameter file (contains dataset paths)

    dim = vox_dim  # <-- dimension for CAD voxelization
    for f in glob.glob(params["shapenet"] + "/**/*/models/model_normalized.obj"):
        catid_cad = f.split("/", 6)[4]
        id_cad = f.split("/", 6)[5]

        outdir = params["shapenet_voxelized"] + "/" + catid_cad
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
        outfile_df = outdir + "/" + id_cad + "__0__.df"

        # -> voxelize as DF
        try:
            program = ["../dfgen/main", f, str(dim), "0", "1", outfile_df]
            print(" ".join(str(x) for x in program))
            subprocess.check_call(program)
        except subprocess.CalledProcessError:
            pass
        # <-

        # -> visualize as PLY file
        try:
            outfile_ply = outfile_df.rsplit(".", 1)[0] + ".ply"
            program = ["../vox2mesh/main", "--in", outfile_df, "--out", outfile_ply, "--is_unitless", "1"]
            print(" ".join(str(x) for x in program))
            subprocess.check_call(program)
        except subprocess.CalledProcessError:
            pass
        # <-
