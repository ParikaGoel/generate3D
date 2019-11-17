#include <dfgen.h>
#include <vox2mesh.h>

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

    if(dfgen(infile,dim,padding,is_unit_cube,vox_file) != 0){
        std::cout<<"Unable to voxelize the given model\n";
    }

    if(vox2mesh(vox_file, ply_file, true) != 0){
        std::cout<<"Unable to convert voxelized model to mesh\n";
    }

    /*mesh object;
    voxel vox;
    load_mesh("/home/parika/WorkingDir/data/ShapeNetCore.v2/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/models/model_normalized.obj",object);
    int rc = voxelize_mesh(vox, object,32,1,true,"/home/parika/WorkingDir/data/ShapeNetCore.v2/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/model_normalized.vox");
    */return 0;
}
