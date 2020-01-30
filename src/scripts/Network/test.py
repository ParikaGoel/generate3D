import torch
import losses
import eval_metric
import numpy as np
import dataset_loader as loader

if __name__ == '__main__':
    output = torch.Tensor([[0.1, 0.5], [0.4, 0.1]])
    target = torch.Tensor([[0, 1], [0, 1]])

    intersection = losses.create_target_mask3(output, target, 4)
