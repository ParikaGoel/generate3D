import torch
import numpy as np


# if __name__ == '__main__':
#     # occ = torch.tensor([24, 36, 57, 45, -1], dtype=torch.float)
#     # index_map = torch.tensor([-1, 1, 2, 1, 0, -1], dtype=torch.long)
#     # grad = torch.tensor([2, 3, 5, 6, 2, 1], dtype=torch.float)
#
#     # one occ one view
#     # index_map_mask = torch.eq(index_map, -1)
#     # index_map[index_map_mask] = occ.size(0)-1
#     # indices, indices_count = torch.unique(index_map, return_counts=True)
#     # tmp = occ.clone().fill_(0)
#     # tmp.index_add_(0, index_map, grad)
#     # tmp[indices] = tmp[indices] / indices_count
#     # mask = torch.eq(tmp, 0.0)
#     # tmp[mask] = occ[mask]
#     # print(tmp)
#
#     # one occ n views
#     # occ = torch.tensor([24, 36, 57, 45], dtype=torch.float)
#     # index_map = torch.tensor([[-1, 1, 2, 1, 0, -1],
#     #                           [-1, 1, 2, 1, 0, -1]], dtype=torch.long)
#     # grad = torch.tensor([[2, 3, 5, 6, 2, 1],
#     #                      [2, 3, 5, 6, 2, 1]], dtype=torch.float)
#     #
#     # tmp = occ.new_empty(size=[occ.size(0)*2+1]).fill_(0)
#     # index_map_mask = torch.eq(index_map, -1)
#     # index_map = torch.stack([torch.add(index_map[i, :], i * occ.size(0)) for i in range(index_map.size(0))])
#     # index_map[index_map_mask] = tmp.size(0) - 1
#     # index_map = torch.flatten(index_map)
#     # indices, indices_count = torch.unique(index_map, return_counts=True)
#     # grad = torch.flatten(grad)
#     # tmp.index_add_(0, index_map, grad)
#     # tmp[indices] = tmp[indices] / indices_count
#     # tmp = tmp[:-1].reshape((2, -1))
#     # mask = torch.eq(tmp, 0.0)
#     # tmp[mask] = occ.repeat(2, 1)[mask]
#     # tmp = torch.mean(tmp, dim=0)
#     # print(tmp)
#
#     # n occ n views
#     n_views = 2
#     batch_size = 2
#     grid_size = 4
#     occ = torch.tensor([[24, 36, 57, 45],
#                          [34, 67, 17, 15]], dtype=torch.float)
#     index_map = torch.tensor([[[-1, 1, 2, 1, 0, -1],
#                               [-1, 1, 2, 1, 0, -1]],
#                               [[-1, 1, 2, 1, 0, -1],
#                               [-1, 1, 2, 1, 0, -1]]], dtype=torch.long)
#     grad = torch.tensor([[[2, 3, 5, 6, 2, 1],
#                          [2, 3, 5, 6, 2, 1]],
#                          [[2, 3, 5, 6, 2, 1],
#                          [2, 3, 5, 6, 2, 1]]], dtype=torch.float)
#
#     tmp = occ.new_empty(size=[batch_size, grid_size * n_views + 1]).fill_(0)
#     index_map_mask = torch.eq(index_map, -1)
#     index_map = torch.stack([torch.stack([torch.add(index_map[b, i, :], i * grid_size) for i in range(index_map[b].size(0))])
#                              for b in range(batch_size)])
#     index_map[index_map_mask] = tmp.size(1) - 1
#     index_map = torch.flatten(index_map, start_dim=1, end_dim=-1)
#     indices = torch.stack([torch.unique(index_map[b]) for b in range(batch_size)])
#     indices_count = torch.stack([torch.unique(index_map[b], return_counts=True)[1] for b in range(batch_size)])
#     grad = torch.flatten(grad, start_dim=1, end_dim=-1)
#     tmp = torch.stack([tmp[b].index_add_(0, index_map[b], grad[b]) for b in range(batch_size)])
#     for b in range(batch_size):
#         tmp[b, indices[b]] = tmp[b, indices[b]] / indices_count[b]
#     tmp = tmp[:,:-1].reshape((batch_size, n_views, -1))
#     mask = torch.eq(tmp, 0.0)
#     occ = torch.stack([occ[b].repeat(2, 1) for b in range(batch_size)])
#     tmp[mask] = occ[mask]
#     tmp = torch.stack([torch.mean(tmp[b], dim=0) for b in range(batch_size)])
#     print(tmp)


if __name__ == '__main__':
    grad = np.loadtxt('grad_out.txt')
    print(grad)