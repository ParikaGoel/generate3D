# shapeComplete

------------------------ To generate the voxelized mesh of shapenet models ----
main.cpp
make command
command : ./bin/complete3D /home/parika/WorkingDir/complete3D/data/02828884/1a40eaf5919b1b3f3eaa2b95b99dae6/models/model_normalized.obj 32 1 1

------------------------ Rendering images from object file -------------------
Run renderer.py. Provide the path of the obj file and the output folder where
you wish to save the rendered images. Also provide how many images you want to render

----------------------- Generating raycasted voxels --------------------------
We need initial raycasted voxels to provide as input to the model. 
For that run raycasting.py. Check the main function of this python file and provide 
all the required paths

----------------------- Training the model -----------------------------------
model.py contains the model. trainer.py handles training of the model