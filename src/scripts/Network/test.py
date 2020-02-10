import torch

if __name__=='__main__':
    # t = torch.empty([2, 2, 2], dtype=torch.int16).fill_(256)
    t = torch.arange(0, 8).reshape((2,2,2)).unsqueeze(0)
    t = t.repeat(3,1,1,1)
    print(t)
    t = torch.flatten(t, start_dim=1)
    print(t)

