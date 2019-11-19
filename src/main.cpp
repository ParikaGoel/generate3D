#include <dfgen.h>
#include <vox2mesh.h>
#include <image.h>

#include <SE3.h>

/*
 * Command :
./bin/complete3D /home/parika/WorkingDir/complete3D/data/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/models/model_normalized.obj 32 1 1
*/

int main(int argc, char *argv[]){
    std::string infile(argv[1]);

    std::stringstream arg2(argv[2]);
    int dim;
    arg2 >> dim;

    std::stringstream arg3(argv[3]);
    int padding;
    arg3 >> padding;

    std::stringstream arg4(argv[4]);
    int is_unit_cube;
    arg3 >> is_unit_cube;

    std::string vox_file = "/home/parika/WorkingDir/complete3D/results/model.vox";
    std::string ply_file = "/home/parika/WorkingDir/complete3D/results/model.ply";

    voxel vox;

    if(dfgen(infile,dim,padding,1,vox_file, vox) != 0){
        std::cout<<"Unable to voxelize the given model\n";
    }

    if(vox2mesh(vox_file, ply_file, true) != 0){
        std::cout<<"Unable to convert voxelized model to mesh\n";
    }

    // Hard coding the camera intrinsics for now -> ToDo(Parika) : Make a parameter file
    Eigen::Matrix3f camera_intr(Eigen::Matrix3f::Identity());
    camera_intr(0,0) = 2.0f;
    camera_intr(1,1) = 2.0f;
    camera_intr(0,2) = 320.0f;
    camera_intr(1,2) = 240.0f;

    // Hard coding the camera pose for now -> ToDo(Parika) : Make a parameter file
    Eigen::Vector3f world_origin(5.0f, 5.0f , 5.0f);
    Eigen::Quaternionf rot(0.7381445,0, 0.4770444, -0.4770444);

    image img(640,480,camera_intr);
    img.setWorld2CamTransform(rot, world_origin);

    img.project(vox);
    img.printDepthValues();

    return 0;
}
