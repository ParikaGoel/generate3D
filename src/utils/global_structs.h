//
// Created by parika on 14.11.19.
//

#ifndef COMPLETE3D_GLOBAL_STRUCTS_H
#define COMPLETE3D_GLOBAL_STRUCTS_H

#include <vector>
#include <Eigen/Dense>

struct vertex{
    float vx;
    float vy;
    float vz;
};

struct face{
    size_t num_vertices;
    std::vector<size_t> vertex_indices;
};

struct mesh{
    std::vector<vertex> vertices;
    std::vector<face> faces;
};

struct voxel{
    std::vector<size_t> dims; // width, height, depth
    Eigen::Vector3f bottom_left;
    Eigen::Vector3f top_right;
    float cell_dist;
    std::vector<bool> occ_grid_vals;
};

#endif //COMPLETE3D_GLOBAL_STRUCTS_H
