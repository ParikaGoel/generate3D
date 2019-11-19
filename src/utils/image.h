//
// Created by parika on 18.11.19.
//

#ifndef COMPLETE3D_IMAGE_H
#define COMPLETE3D_IMAGE_H

#include <vector>

struct image{
    size_t width;
    size_t height;
    std::vector<float> depth;
    image(size_t w,  size_t h){
        width = w;
        height = h;
        depth.resize(w*h);
    }
};

#endif //COMPLETE3D_IMAGE_H
