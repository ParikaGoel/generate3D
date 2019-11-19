//
// Created by parika on 18.11.19.
//

#ifndef COMPLETE3D_CAMERA_H
#define COMPLETE3D_CAMERA_H

#include <image.h>
#include <voxel_loader.h>

struct camera{
    float fx;
    float fy;
    float cx;
    float cy;
};

void project(struct camera& cam, struct image& img, struct voxel& vox);
void unproject(struct camera& cam);

#endif //COMPLETE3D_CAMERA_H
