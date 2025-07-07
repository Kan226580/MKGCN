import os
import gzip
import torch
from torch_geometric.data import InMemoryDataset, Data


class HiggsTwitter(InMemoryDataset):
    URL_MAP = {
        'RT': 'http://snap.stanford.edu/data/higgs-retweet_network.edgelist.gz',
        'MT': 'http://snap.stanford.edu/data/higgs-mention_network.edgelist.gz',
        'RE': 'http://snap.stanford.edu/data/higgs-reply_network.edgelist.gz'
    }

    FILE_MAP = {
        'RT': 'retweet_network.edgelist.gz',
        'MT': 'mention_network.edgelist.gz',
        'RE': 'reply_network.edgelist.gz'
    }

    def __init__(self, root, net_type='RT', transform=None, pre_transform=None):
        self.net_type = net_type.upper()
        assert self.net_type in ['RT', 'MT', 'RE'], "net_type must be 'RT', 'MT' or 'RE'"

        self.root = os.path.join(root, self.net_type)
        os.makedirs(self.root, exist_ok=True)

        super().__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:

        return os.path.join(self.root, 'raw')

    @property
    def processed_dir(self) -> str:

        return os.path.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        return [self.FILE_MAP[self.net_type]]

    @property
    def processed_file_names(self):
        return [f'higgs_{self.net_type.lower()}_data.pt']

    def download(self):
        from six.moves import urllib

        os.makedirs(self.raw_dir, exist_ok=True)

        url = self.URL_MAP[self.net_type]
        path = os.path.join(self.raw_dir, self.raw_file_names[0])

        if not os.path.exists(path):
            print(f'Downloading {self.net_type} network from {url} to {path}...')
            urllib.request.urlretrieve(url, path)
        else:
            print(f'File already exists at {path}')

    def process(self):
        gz_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        edges = []
        node_set = set()

        with gzip.open(gz_path, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                src, dst = int(parts[0]), int(parts[1])
                edges.append((src, dst))
                node_set.update([src, dst])

        node_list = sorted(node_set)
        node_map = {node: i for i, node in enumerate(node_list)}
        num_nodes = len(node_map)

        edge_index = []
        for src, dst in edges:
            edge_index.append([node_map[src], node_map[dst]])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        feature_dim = 256
        x = torch.randn(num_nodes, feature_dim)

        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        os.makedirs(self.processed_dir, exist_ok=True)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'{self.__class__.__name__}({self.net_type})'
