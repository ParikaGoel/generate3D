import torch
import numpy as np
from PIL import Image
import dataset_loader as loader


if __name__ == '__main__':
    predicted = "/home/parika/WorkingDir/complete3D/Assets_remote/output-network/04379243/predicted_test_output/best_model.npy"
    gt = "/home/parika/WorkingDir/complete3D/Assets_remote/output-network/04379243/predicted_test_output/gt.npy"

    color = np.array([169, 169, 0])
    loader.df_to_mesh(predicted, 4.0, color)
    loader.df_to_mesh(gt)