//
// Created by parika on 17.11.19.
//

#ifndef COMPLETE3D_DFGEN_H
#define COMPLETE3D_DFGEN_H

#include <string>
#include <voxel_loader.h>

int dfgen(std::string infile,
        int dim, int padding,
        int is_unit_cube, std::string outfile, voxel& vox);

#endif //COMPLETE3D_DFGEN_H
