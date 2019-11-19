//
// Created by parika on 18.11.19.
//

#ifndef COMPLETE3D_IMAGE_H
#define COMPLETE3D_IMAGE_H

#include <vector>
    #include <Eigen/Dense>

#include <vox2mesh.h>

class image{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    image(size_t h, size_t w, Eigen::Matrix3f& intrinsics);

    const size_t getWidth() const;
    const size_t getHeight() const;
    const Eigen::Matrix4f& getWorld2CamTransform() const;

    void setWorld2CamTransform(const Eigen::Matrix4f& pose);
    void setWorld2CamTransform(const Eigen::Quaternionf& world_axes_orientation, const Eigen::Vector3f& world_origin);

    void project(voxel& vox);
    void project(const std::vector<Eigen::Vector3f>& global_coords);
    Eigen::Vector3f transformIntoCameraCoord(const Eigen::Vector3f& global_coord);
    Eigen::Vector2i projectOntoImagePlane(const Eigen::Vector3f& camera_coord);
    bool contains(const Eigen::Vector2i& img_coord);

    void printDepthValues();

private:

    const size_t width_; // number of columns or y-values
    const size_t height_; // number of rows or x-values
    std::vector<float> depth_map_;
    std::vector<Eigen::Vector3f> points_3d_;
    std::vector<Eigen::Vector3f> points_3d_global_;
    Eigen::Matrix3f intrinsics_matrix_; // camera intrinsics matrix
    // define the position of the camera center and the camera's heading in world coordinates
    // transform_world_to_cam_ = [R|t] where t = RC
    // C is the position of the origin of the world coordinate system expressed in
    // coordinates of the camera-centered coordinate system
    // this will give transformation from world system to camera system
    Eigen::Matrix4f transform_world_to_cam_;
};

#endif //COMPLETE3D_IMAGE_H
