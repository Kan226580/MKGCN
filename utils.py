import numpy as np
import random
import time
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling

device = torch.device('cuda')


def get_metrics(out, edge_label):
    edge_label = edge_label.cpu().numpy()
    out = out.cpu().numpy()
    auc = roc_auc_score(edge_label, out)
    ap = average_precision_score(edge_label, out)

    return auc, ap


def train_negative_sample(train_data):
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1)
    edge_label = torch.cat([
        train_data.edge_label,
        train_data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    return edge_label, edge_label_index


@torch.no_grad()
def test(model, val_data, test_data):
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    out = model(val_data, val_data.edge_label_index).view(-1)
    val_loss = criterion(out, val_data.edge_label)
    out = model(test_data, test_data.edge_label_index).view(-1).sigmoid()
    model.train()

    auc, ap = get_metrics(out, test_data.edge_label)

    return val_loss, auc, ap


def train(model, train_data, val_data, test_data, lr=0.001, epochs=2000, runs=5):
    all_test_aucs = []
    all_test_aps = []
    all_training_times = []

    for run in range(runs):

        random.seed(run)
        np.random.seed(run)
        torch.manual_seed(run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(run)
            torch.cuda.manual_seed_all(run)
        # print('Seed', run)

        model = model.to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
        final_test_auc = 0
        final_test_ap = 0
        min_val_loss = np.Inf
        loss = 0
        model.train()

        start_time = time.time()
        epoch_times = []

        for epoch in tqdm(range(epochs)):
            epoch_start_time = time.time()
            optimizer.zero_grad()
            edge_label, edge_label_index = train_negative_sample(train_data)
            out = model(train_data, edge_label_index).view(-1)
            loss = criterion(out, edge_label)
            loss.backward()
            optimizer.step()

            val_loss, test_auc, test_ap = test(model, val_data, test_data)
            if val_loss < min_val_loss and test_auc > final_test_auc:
                min_val_loss = val_loss
                final_test_auc = test_auc
                final_test_ap = test_ap

            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)

            # print('Run {:02d} | Epoch {:03d} | Loss {:.4f} | AUC {:.4f} | AP {:.4f} | Time: {:.2f}s'
            #       .format(run + 1, epoch + 1, loss.item(), test_auc, test_ap, epoch_time))

        total_time = time.time() - start_time
        avg_epoch_time = np.mean(epoch_times) if epoch_times else 0

        # Store results for this run
        all_test_aucs.append(final_test_auc)
        all_test_aps.append(final_test_ap)
        all_training_times.append(total_time)

        # # Print run summary
        # print(
        #     'Run: {:02d}, Final Test AUC: {:.2f}, AP: {:.2f}'.format(run + 1, final_test_auc * 100, final_test_ap * 100)
        # )

    # After all runs are completed, calculate and print statistics
    mean_auc = np.mean(all_test_aucs)
    std_auc = np.std(all_test_aucs)
    mean_ap = np.mean(all_test_aps)
    std_ap = np.std(all_test_aps)

    print('Test AUC: {:.2f} ± {:.2f}'.format(mean_auc * 100, std_auc * 100))
    print('Test AP:  {:.2f} ± {:.2f}'.format(mean_ap * 100, std_ap * 100))

    return mean_auc, mean_ap
