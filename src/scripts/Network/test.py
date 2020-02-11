import torch

if __name__=='__main__':
    input = torch.tensor([12, 56, 45, 37], dtype=torch.float)
    input = torch.cat([input, torch.tensor([-1], dtype=torch.float)])
    print(input)
    map = torch.tensor([-1, 1, 2, 1, 3, -1])

    grad_out = torch.tensor([4, 2, 5, 3, 6, 7])
    input[map] = 0
    input[map] = grad_out
    print(input)


