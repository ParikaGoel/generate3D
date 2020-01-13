# shapeComplete

## Get started

1. Clone repo:

```git clone https://github.com/skanti/Scan2CAD.git```

2. Ask for dataset: (You will need *ShapeNet*). 

3. Copy dataset content into `./Assets/`.

4. Compile `c++` programs

```
cd {vox2mesh, dfgen}
make
```

5. Voxelize CADs (shapenet):

```python3 ./src/scripts/CADVoxelization.py```

6. Rendering images of the shapenet model from different viewpoints

``` python3 ./src/scripts/renderer.py ```

7. Generating raycasted voxels for input to the network
(For now the model id and category id are hardcoded in python file)

``` python3 ./src/scripts/raytracing.py ```

8. Training the network
(For now the model id and category id are hardcoded in python file)

``` python3 ./src/scripts/trainer.py ```
