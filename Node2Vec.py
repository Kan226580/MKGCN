from sklearn.linear_model import LogisticRegression
import torch_geometric.transforms as T
from tqdm import tqdm
import torch
from torch_geometric.nn import Node2Vec
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.datasets import Planetoid, CitationFull, WebKB, Amazon, AttributedGraphDataset
import numpy as np
from Higgs import HiggsTwitter

root="./data"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      disjoint_train_ratio=0),
])


def Run(name):
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
    train_data, val_data, test_data = dataset[0]

    all_auc = []
    all_ap = []

    for run in range(5):
        import random
        random.seed(run)
        np.random.seed(run)
        torch.manual_seed(run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(run)
            torch.cuda.manual_seed_all(run)

        model = Node2Vec(
            train_data.edge_index, embedding_dim=128, walk_length=20,
            context_size=10, walks_per_node=10,
            num_negative_samples=1, p=1, q=1, sparse=True, num_nodes=train_data.num_nodes).to(device)
        model.reset_parameters()

        num_workers = 0

        loader = model.loader(batch_size=2048, shuffle=True, num_workers=num_workers)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

        model.train()
        for epoch in tqdm(range(500)):
            total_loss = 0
            for i, (pos_rw, neg_rw) in enumerate(loader):
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # print('Epoch {:03d} Loss {:.4f}'.format(epoch, total_loss / len(loader)))

        torch.save(model.embedding.weight.data.cpu(), f'./working/node2vec_embedding_{name}.pt')

        z = torch.load(root + f'/node2vec_embedding_{name}.pt')

        train_src = z[train_data.edge_label_index[0].cpu()]
        train_dst = z[train_data.edge_label_index[1].cpu()]
        train_x = torch.cat([train_src, train_dst], dim=-1).numpy()
        train_y = train_data.edge_label.cpu().numpy()

        val_src = z[val_data.edge_label_index[0].cpu()]
        val_dst = z[val_data.edge_label_index[1].cpu()]
        val_x = torch.cat([val_src, val_dst], dim=-1).numpy()
        val_y = val_data.edge_label.cpu().numpy()

        test_src = z[test_data.edge_label_index[0].cpu()]
        test_dst = z[test_data.edge_label_index[1].cpu()]
        test_x = torch.cat([test_src, test_dst], dim=-1).numpy()
        test_y = test_data.edge_label.cpu().numpy()

        clf = LogisticRegression()
        clf.fit(train_x, train_y)
        score = clf.predict_proba(test_x)[:, -1]

        test_auc = roc_auc_score(test_y, score)
        test_ap = average_precision_score(test_y, score)

        print('Run: {:02d}, Final Test AUC: {:.2f}, AP: {:.2f}'.format(run + 1, test_auc * 100, test_ap * 100))
        all_auc.append(test_auc)
        all_ap.append(test_ap)

    mean_auc = np.mean(all_auc)
    std_auc = np.std(all_auc)
    mean_ap = np.mean(all_ap)
    std_ap = np.std(all_ap)

    print('Test AUC: {:.2f} ± {:.2f}'.format(mean_auc * 100, std_auc * 100))
    print('Test AP:  {:.2f} ± {:.2f}'.format(mean_ap * 100, std_ap * 100))


for name in ['Wiki', 'Facebook']:
    print(name)
    Run(name)


