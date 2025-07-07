import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from FastKANLayer import FastKANLayer


# class MKGCN(nn.Module):
#     def __init__(self, in_chnl, hid_chnl, out_chnl, n_grid, num_works=3):
#         super().__init__()
#         assert len(hid_chnl) == 3, "Exactly 3 paths required"
#         assert num_works >= 3, "Need at least 3 devices for parallelization"

#         self.paths = nn.ModuleList()
#         for i, hid in enumerate(hid_chnl):

#             device = torch.device(f'cuda:{i % num_works}')
#             path = nn.ModuleList([
#                 GCNConv(in_chnl, hid).to(device),
#                 GCNConv(hid, hid).to(device),
#                 FastKANLayer(hid, out_chnl, n_grid).to(device)
#             ])
#             self.paths.append(path)

#         self.num_paths = len(hid_chnl)
#         self.num_works = num_works

#     def encode(self, data):
#         x, edge_index = data.x, data.edge_index
#         path_outputs = []

#         futures = []
#         for i, path in enumerate(self.paths):
#             device = torch.device(f'cuda:{i % self.num_works}')
#             x_i = x.to(device)
#             edge_index_i = edge_index.to(device)
#             futures.append(torch.jit.fork(
#                 self._compute_path, path, x_i, edge_index_i))

#         for future in futures:
#             path_outputs.append(torch.jit.wait(future).to('cuda:0'))

#         return torch.stack(path_outputs, dim=0).mean(dim=0)

#     def _compute_path(self, path, x, edge_index):
#         h = path[0](x, edge_index)
#         h = path[1](h, edge_index)
#         return path[2](h)

#     def decode(self, z, edge_label_index):
#         src = z[edge_label_index[0]]
#         dst = z[edge_label_index[1]]
#         return (src * dst).sum(dim=-1)

#     def forward(self, data, edge_label_index):
#         z = self.encode(data)
#         return self.decode(z, edge_label_index)

class MKGCN(nn.Module):
    def __init__(self, in_chnl, hid_chnl, out_chnl, n_grid):
        super(MKGCN, self).__init__()

        self.paths = nn.ModuleList()
        for hid_chnl in hid_chnl:
            path = nn.ModuleList([
                GCNConv(in_chnl, hid_chnl),
                GCNConv(hid_chnl, hid_chnl),
                FastKANLayer(hid_chnl, out_chnl, n_grid)
            ])
            self.paths.append(path)

    def reset_parameters(self):

        for path in self.paths:
            for layer in path:
                layer.reset_parameters()

    def encode(self, data):
        x, edge_index = data.x, data.edge_index

        path_outputs = []

        for path in self.paths:
            h = path[0](x, edge_index)
            h = path[1](h, edge_index)
            h = path[2](h)

            path_outputs.append(h)

        return torch.stack(path_outputs, dim=0).mean(dim=0)

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def forward(self, data, edge_label_index):
        z = self.encode(data)
        return self.decode(z, edge_label_index)

class MMGCN(nn.Module):
    def __init__(self, in_chnl, hid_chnl, out_chnl, n_grid):
        super(MMGCN, self).__init__()

        self.paths = nn.ModuleList()
        for hid_chnl in hid_chnl:
            path = nn.ModuleList([
                GCNConv(in_chnl, hid_chnl),
                GCNConv(hid_chnl, hid_chnl),
                nn.Linear(hid_chnl, hid_chnl),
                nn.Linear(hid_chnl, out_chnl)
            ])
            self.paths.append(path)

    def reset_parameters(self):

        for path in self.paths:
            for layer in path:
                layer.reset_parameters()

    def encode(self, data):
        x, edge_index = data.x, data.edge_index

        path_outputs = []

        for path in self.paths:
            h = path[0](x, edge_index)
            h = path[1](h, edge_index)
            h = path[2](h).relu()
            h = path[3](h)

            path_outputs.append(h)

        return torch.stack(path_outputs, dim=0).mean(dim=0)

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def forward(self, data, edge_label_index):
        z = self.encode(data)
        return self.decode(z, edge_label_index)

class A_MKGCN(nn.Module):
    def __init__(self, in_chnl, hid_chnl, out_chnl, n_grid):
        super(A_MKGCN, self).__init__()

        self.n_paths = len(hid_chnl)

        self.paths = nn.ModuleList()
        for hid in hid_chnl:
            path = nn.ModuleList([
                GCNConv(in_chnl, hid),
                GCNConv(hid, hid),
                FastKANLayer(hid, out_chnl, n_grid)
            ])
            self.paths.append(path)

        self.att_weights = nn.Parameter(torch.ones(self.n_paths) / self.n_paths)

    def reset_parameters(self):

        for path in self.paths:
            for layer in path:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

        nn.init.constant_(self.att_weights, 1.0 / self.n_paths)

    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        path_outputs = []

        for path in self.paths:
            h = path[0](x, edge_index)
            h = path[1](h, edge_index)
            h = path[2](h)
            path_outputs.append(h)

        weights = F.softmax(self.att_weights, dim=0)

        weighted_output = torch.zeros_like(path_outputs[0])
        for i in range(self.n_paths):
            weighted_output += weights[i] * path_outputs[i]

        return weighted_output

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)

    def forward(self, data, edge_label_index):
        z = self.encode(data)
        return self.decode(z, edge_label_index)