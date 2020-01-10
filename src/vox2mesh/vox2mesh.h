//
// Created by parika on 17.11.19.
//

#ifndef COMPLETE3D_VOX2MESH_H
#define COMPLETE3D_VOX2MESH_H

#include <Eigen/Dense>
#include <voxel_loader.h>

struct PlyMesh {
    Eigen::Matrix<float, -1, -1> V;
    Eigen::Matrix<float, -1, -1> N;
    Eigen::Matrix<uint8_t, -1, -1> C;
    Eigen::Matrix<uint32_t, -1, -1> F;
};

void get_position_and_color_from_vox(
        voxel &vox, PlyMesh &mesh, Eigen::Vector3f voxelsize,
        float trunc = 1.0f, std::string cmap = "jet");

void write_ply(const std::string & filename, PlyMesh &mesh);

int vox2mesh(std::string vox_file,
             std::string ply_file,
             std::string txt_file,
             bool is_unitless = false,
             bool redcenter = false,
             std::string cmap = "jet",
             float trunc = 1.0f);

#endif //COMPLETE3D_VOX2MESH_H
