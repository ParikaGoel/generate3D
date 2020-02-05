import torch

if __name__=='__main__':
    t = torch.tensor([[0,1],
                       [2,3]])
    t = torch.flatten(t)
    print(t)