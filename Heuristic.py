import numpy as np
import random
import torch
import torch_geometric.transforms as T
from Higgs import HiggsTwitter
from torch_geometric.datasets import Planetoid, CitationFull, WebKB, Amazon, AttributedGraphDataset
from sklearn.metrics import roc_auc_score, average_precision_score


def get_roc_score(edges_pos, edges_neg, score_matrix):
    preds_pos = []
    for edge in edges_pos:
        preds_pos.append(score_matrix[edge[0], edge[1]])

    preds_neg = []
    for edge in edges_neg:
        preds_neg.append(score_matrix[edge[0], edge[1]])

    preds_all = np.hstack([preds_pos, preds_neg])
    labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])

    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score


class HeuristicPredictor:
    def __init__(self, adj_sparse, device):
        self.device = device
        self.adj_sparse = adj_sparse.to(device)
        self.shape = adj_sparse.shape
        self.col = torch.sparse.sum(adj_sparse, dim=0).to_dense().view(-1, 1)
        self.I = torch.ones(self.shape[0], 1, device=self.device)
        self.epsilon = 1e-10  # Small value to avoid division by zero

    def CN(self):
        return torch.sparse.mm(self.adj_sparse, self.adj_sparse)

    def PA(self):
        return torch.sparse.mm(self.col, self.col.t())

    def RA(self):
        col_inv = 1 / (self.col + self.epsilon)
        return torch.sparse.mm(self.adj_sparse.multiply(col_inv.t()), self.adj_sparse)

    def AA(self):
        col_log_inv = 1 / (torch.log(self.col + 1) + self.epsilon)  # +1 to avoid log(0)
        return torch.sparse.mm(self.adj_sparse.multiply(col_log_inv.t()), self.adj_sparse)

    def JI(self):
        CN = torch.sparse.mm(self.adj_sparse, self.adj_sparse)
        denominator = torch.sparse.mm(self.I, self.col.t()) + \
                      torch.sparse.mm(self.I, self.col.t()).t() - CN
        # Add epsilon to avoid division by zero
        denominator = denominator + self.epsilon
        return CN * (1 / denominator)

    def Sorensen(self):
        CN = torch.sparse.mm(self.adj_sparse, self.adj_sparse)
        denominator = torch.sparse.mm(self.I, self.col.t()) + \
                      torch.sparse.mm(self.I, self.col.t()).t()
        # Add epsilon to avoid division by zero
        denominator = denominator + self.epsilon
        return CN * (2 / denominator)

    def salton(self):
        CN = torch.sparse.mm(self.adj_sparse, self.adj_sparse)
        col_sqrt = torch.sqrt(self.col + self.epsilon)
        denominator = torch.sparse.mm(col_sqrt, col_sqrt.t())
        # Add epsilon to avoid division by zero
        denominator = denominator + self.epsilon
        return CN * (1 / denominator)

    def compute_all(self):
        return [
            # self.CN(),
            self.PA(),
            # self.RA(),
            # self.AA(),
            self.JI(),
            # self.Sorensen(),
            # self.salton()
        ]


def Run(name):
    all_PA_roc = []
    all_PA_ap = []
    all_JI_roc = []
    all_JI_ap = []
    root="./data"

    for run in range(5):
        random.seed(run)
        np.random.seed(run)
        torch.manual_seed(run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(run)
            torch.cuda.manual_seed_all(run)
        if name in {'Cora', 'CiteSeer', 'PubMed'}:
            dataset = Planetoid(root, name)
        elif name in {'RT', 'MT', 'RE'}:
            dataset = HiggsTwitter(root, name)
        elif name == 'DBLP':
            dataset = CitationFull(root, name)
        elif name in {'Cornell', 'Texas', 'Wisconsin'}:
            dataset = WebKB(root, name)
        elif name in {'Computers', 'Photo'}:
            dataset = Amazon(root, name)
        elif name in {'Wiki', 'Facebook'}:
            dataset = AttributedGraphDataset(root, name)

        data = dataset[0]

        split = T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                                  add_negative_train_samples=False)
        train_data, val_data, test_data = split(data)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        num_nodes = data.num_nodes
        edge_index = train_data.edge_index.to(device)
        values = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
        train_adj_sparse = torch.sparse_coo_tensor(
            edge_index, values, (num_nodes, num_nodes)
        ).coalesce()

        heuristic = HeuristicPredictor(train_adj_sparse, device)
        score_matrices = heuristic.compute_all()
        model_names = ['PA', 'JI']

        val_edges_pos = val_data.edge_label_index[:, val_data.edge_label == 1].t().cpu().numpy()
        val_edges_neg = val_data.edge_label_index[:, val_data.edge_label == 0].t().cpu().numpy()

        test_edges_pos = test_data.edge_label_index[:, test_data.edge_label == 1].t().cpu().numpy()
        test_edges_neg = test_data.edge_label_index[:, test_data.edge_label == 0].t().cpu().numpy()

        # print("Validation Set Results:")
        # for name, scores in zip(model_names, score_matrices):
        #     scores_cpu = scores.to_dense().cpu().detach().numpy()
        #     roc, ap = get_roc_score(val_edges_pos, val_edges_neg, scores_cpu)
        #     roc, ap = roc * 100, ap * 100
        #     print(f"{name}: ROC-AUC = {roc:.2f}, AP = {ap:.2f}")

        print("\nTest Set Results:")
        for name, scores in zip(model_names, score_matrices):
            scores_cpu = scores.to_dense().cpu().detach().numpy()
            roc, ap = get_roc_score(test_edges_pos, test_edges_neg, scores_cpu)
            roc, ap = roc * 100, ap * 100
            print(f"{name}: ROC-AUC = {roc:.2f}, AP = {ap:.2f}")
            if name == 'PA':
                all_PA_roc.append(roc)
                all_PA_ap.append(ap)
            elif name == 'JI':
                all_JI_roc.append(roc)
                all_JI_ap.append(ap)

    mean_PA_auc = np.mean(all_PA_roc)
    std_PA_auc = np.std(all_PA_roc)
    mean_PA_ap = np.mean(all_PA_ap)
    std_PA_ap = np.std(all_PA_ap)

    print('Test PA AUC: {:.2f} ± {:.2f}'.format(mean_PA_auc, std_PA_auc))
    print('Test PA AP:  {:.2f} ± {:.2f}'.format(mean_PA_ap, std_PA_ap))

    mean_JI_auc = np.mean(all_JI_roc)
    std_JI_auc = np.std(all_JI_roc)
    mean_JI_ap = np.mean(all_JI_ap)
    std_JI_ap = np.std(all_JI_ap)

    print('Test JI AUC: {:.2f} ± {:.2f}'.format(mean_JI_auc, std_JI_auc))
    print('Test JI AP:  {:.2f} ± {:.2f}'.format(mean_JI_ap, std_JI_ap))
    print('==============================================')


for name in [
    'Cora', 'CiteSeer', 'PubMed',
    'DBLP',
    'RT', 'MT', 'RE',
    'Computers', 'Photo',
    'Wiki', 'Facebook'
]:
    print(name)
    Run(name)
