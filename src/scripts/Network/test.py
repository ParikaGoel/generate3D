import torch

if __name__=='__main__':
    vertex_coords = torch.tensor([[[22, 23],
                                  [25, 26]],
                                 [[24, 27],
                                  [13, 29]]])

    min = torch.min(vertex_coords, dim=0)

    print(min)

