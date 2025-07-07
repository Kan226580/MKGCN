import torch
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid, CitationFull, WebKB, Amazon, AttributedGraphDataset

from torch_geometric.datasets import Planetoid, CitationFull, WebKB

from Higgs import HiggsTwitter
from MKGCN import MKGCN
from GNNs import GCN, SAGE, GAT
from utils import train

device = torch.device('cuda')


def Run(hid_chnl, out_chnl, n_grid, epochs, name, root, model_name):
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                          add_negative_train_samples=False, disjoint_train_ratio=0)
    ])

    dataset = None

    if name in {'Cora', 'CiteSeer', 'PubMed'}:
        dataset = Planetoid(root, name, transform=transform)
    elif name in {'RT', 'MT', 'RE'}:
        dataset = HiggsTwitter(root, name, transform=transform)
    elif name == 'DBLP':
        dataset = CitationFull(root, name, transform=transform)
    elif name in {'Cornell', 'Texas', 'Wisconsin'}:
        dataset = WebKB(root, name, transform=transform)

    elif name in {'Computers', 'Photo'}:
        dataset = Amazon(root, name, transform=transform)
    elif name in {'Wiki', 'Facebook'}:
        dataset = AttributedGraphDataset(root, name, transform=transform)

    print(f'{name} ready...')
    print(f'{model_name} ready...')

    train_data, val_data, test_data = dataset[0]
    # print(train_data)
    # print(val_data)
    # print(test_data)

    model = None
    lr = None

    if model_name == 'MKGCN':

        model = MKGCN(dataset.num_features, hid_chnl, out_chnl, n_grid).to(device)
        lr = 0.001
        print(f'lr = {lr} grid = {n_grid} hidden = {hid_chnl} output = {out_chnl}')

    elif model_name == 'GCN':

        model = GCN(dataset.num_features, 256, 256, 2).to(device)
        lr = 0.005

    elif model_name == 'SAGE':

        model = SAGE(dataset.num_features, 256, 256, 2).to(device)
        lr = 0.005

    elif model_name == 'GAT':

        model = GAT(dataset.num_features, 256, 256, 2).to(device)
        lr = 0.005


    test_auc, test_ap = train(
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        lr=lr,
        epochs=epochs
    )


for name in [
    'Cora', 'CiteSeer', 'PubMed',
    'DBLP',
    'RT', 'MT', 'RE',
    'Computers', 'Photo',
    'Wiki', 'Facebook'
]:
    for n_grid in [10, 20, 30, 40, 50]:
        Run(
            hid_chnl=[256, 64, 16],
            out_chnl=128,
            n_grid=n_grid,
            epochs=500,
            name=name,
            root="./data",
            model_name='MKGCN'
        )
        print('==================================================')

# for name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'RT', 'MT', 'RE']:
#     for model_name in ['GCN', 'SAGE', 'GAT']:
#         Run(
#             hid_chnl=[8, 32, 128],
#             out_chnl=16,
#             n_grid=1,
#             epochs=500,
#             name=name,
#             root="/kaggle/working/datasets",
#             model_name=model_name
#         )
#         print('==================================================')
