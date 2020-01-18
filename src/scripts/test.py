import torch
from torch import nn

if __name__ == '__main__':
    m = nn.Sigmoid()
    input = torch.randn(2)
    print(input)
    print(m(input))
