# complete3D

## Get started

1. Clone repo:

```git clone https://github.com/ParikaGoel/complete3D```

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
(There are different scripts to train occupancy and distance field representation. Example commands are given below)

``` python3 trainer_occ.py --synset_id 04379243 --model_name Net3D --gt_type occ --train_batch_size 32 --val_batch_size 32 --truncation 3 --lr 0.001 --lr_decay 0.5 --n_vis 20```

``` python3 trainer_df.py --synset_id 04379243 --model_name UNet3D --gt_type tdflog --use_logweight --train_batch_size 8 --val_batch_size 32 --truncation 3 --lr 0.001 --lr_decay 0.1 --n_vis 20 --num_epochs 50```

More parameters can be customized which can be seen in the script
