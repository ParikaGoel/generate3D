//
// Created by parika on 18.11.19.
//

#include <camera.h>
#include <vox2mesh.h>
#include <SE3.h>

void project(struct camera& cam, struct image& img, struct voxel& vox){
    PlyMesh mesh;

    Eigen::Vector3f t;
    Eigen::Quaternionf q;
    Eigen::Vector3f s;
    decompose_mat4(vox.grid2world, t, q, s);

    get_position_and_color_from_vox(vox, mesh, s);

    for(size_t col = 0; col < mesh.V.cols(); col++){
        float x_3d = mesh.V(0,col);
        float y_3d = mesh.V(1,col);
        float z_3d = mesh.V(2,col);
    }
}
