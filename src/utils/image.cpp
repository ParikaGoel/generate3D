//
// Created by parika on 19.11.19.
//

#include <SE3.h>

#include <globals.h>
#include <image.h>

image::image(size_t h, size_t w, Eigen::Matrix3f& intrinsics):
    width_(w), height_(h), intrinsics_matrix_(intrinsics)
{
    depth_map_.resize(width_*height_,MINF);
    transform_world_to_cam_ = Eigen::Matrix4f::Identity();
}

const size_t image::getWidth() const {
    return width_;
}

const size_t image::getHeight() const {
    return height_;
}

const Eigen::Matrix4f& image::getWorld2CamTransform() const{
    return transform_world_to_cam_;
}
void image::setWorld2CamTransform(const Eigen::Matrix4f &pose) {
    transform_world_to_cam_ = pose;
}

void image::setWorld2CamTransform(const Eigen::Quaternionf& world_axes_orientation, const Eigen::Vector3f& world_origin){
    transform_world_to_cam_ = Eigen::Matrix4f::Identity();
    transform_world_to_cam_.block(0, 0, 3, 3) = world_axes_orientation.toRotationMatrix();
    transform_world_to_cam_.block(0, 3, 3, 1) = world_origin;
}

void image::project(voxel& vox){

    std::vector<Eigen::Vector3f> vertices;

    double trunc = 1.0f; // voxel truncation dist
    for (int k = 0; k < vox.dims[2]; k++) {
        for (int j = 0; j < vox.dims[1]; j++) {
            for (int i = 0; i < vox.dims[0]; i++) {
                int index = k*vox.dims[1]*vox.dims[0] + j*vox.dims[0] + i;
                if (std::abs(vox.sdf[index]) <= trunc * vox.res) {
                    Eigen::Vector3f p = (vox.grid2world*Eigen::Vector4f(i, j, k, 1)).topRows(3);
                    vertices.push_back(p);
                }
            }
        }
    }

    std::cout<<"Number of vertices: " << vertices.size() << "\n";
    project(vertices);
}

void image::project(const std::vector<Eigen::Vector3f> &global_coords) {
    size_t nValid = 0;
    for(const auto& point_3d_global : global_coords){
        points_3d_global_.emplace_back(point_3d_global);
        Eigen::Vector3f camera_coord = transformIntoCameraCoord(point_3d_global);
        points_3d_.emplace_back(camera_coord);
        Eigen::Vector2i img_coord = projectOntoImagePlane(camera_coord);

        if(contains(img_coord)){
            nValid++;
            size_t index = img_coord[0] * width_ + img_coord[1];
            depth_map_[index] = camera_coord[2]; // ToDo(Parika) : Assuming the z-value in camera system gives the depth value. Check Again
//            std::cout<<"depth value at " << index << " is " << depth_map_[index] << "\n";
        }
    }

//    std::cout<<"Valid depth values: " << nValid << "\n";
}

Eigen::Vector3f image::transformIntoCameraCoord(const Eigen::Vector3f& global_coord) {
    const auto rotation = transform_world_to_cam_.block(0, 0, 3, 3);
    const auto translation = transform_world_to_cam_.block(0, 3, 3, 1);
    return rotation * global_coord + translation;
}

Eigen::Vector2i image::projectOntoImagePlane(
        const Eigen::Vector3f &camera_coord) {
    Eigen::Vector3f projected = intrinsics_matrix_ * camera_coord;

    if(projected[2] == 0)
        return Eigen::Vector2i(MINF,MINF);

    projected = projected/projected[2];

    return Eigen::Vector2i((int)round(projected[0]),(int)round(projected[1]));
}

bool image::contains(const Eigen::Vector2i &img_coord) {
    return (img_coord[0] >= 0 && img_coord[0] < height_ && img_coord[1] >= 0 && img_coord[1] < width_);
}

void image::printDepthValues() {
    for(size_t i = 0; i < height_*width_ ; i++){
        if(depth_map_[i] == MINF)
            continue;
        std::cout<< i << "  : " << depth_map_[i] << "\n";
    }
}