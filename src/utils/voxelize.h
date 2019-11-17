//
// Created by parika on 15.11.19.
//

#ifndef COMPLETE3D_VOXELIZE_H
#define COMPLETE3D_VOXELIZE_H

#include <Eigen/Dense>
#include <global_structs.h>

inline static void update_minmax(float& point, float& min_box, float& max_box){
    if (point < min_box)
        min_box = point;
    else if(point > max_box)
        max_box = point;
}

inline static void update_minmax(Eigen::Vector3f& point, Eigen::Vector3f& min_box, Eigen::Vector3f& max_box){
    for(size_t i =0 ; i < 3; i++){
        update_minmax(point[i], min_box[i], max_box[i]);
    }
}

bool triangle_box_intersection(const Eigen::Vector3f &min,
        Eigen::Vector3f &max, const Eigen::Vector3f &v1,
        const Eigen::Vector3f &v2, const Eigen::Vector3f &v3);
void voxelize_to_occ_grid(mesh& mesh, voxel& occ_grid);
int voxelize_mesh(voxel& vox, mesh& mesh, size_t dim, size_t padding, bool is_unit_cube, std::string out_file);
void save_vox(std::string filename, voxel &vox);

#endif //COMPLETE3D_VOXELIZE_H
