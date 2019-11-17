#include <mesh_loader.h>
#include <voxelize.h>

int main(){

    mesh object;
    voxel vox;
    load_mesh("/home/parika/WorkingDir/data/ShapeNetCore.v2/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/models/model_normalized.obj",object);
    int rc = voxelize_mesh(vox, object,32,1,true,"/home/parika/WorkingDir/data/ShapeNetCore.v2/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/model_normalized.vox");
    return 0;
}
