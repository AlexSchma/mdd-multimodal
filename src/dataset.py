import numpy as np
import os
import pandas as pd
import torch

from torch_geometric.utils import dense_to_sparse
from torch.utils.data import Dataset


# Dataset for fMRI data that stores everything in memory
class RESTfMRIDataset(Dataset):
    def __init__(
        self,
        metadata_path="./REST-meta-MDD/metadata.csv",
        data_dir="./REST-meta-MDD/fMRI/AAL",
    ):
        super(RESTfMRIDataset, self).__init__()
        self.cache_path = os.path.join("cache", data_dir)

        if self._cache_exists():
            self.edge_indices, self.node_features = self._load_cache()
        else:
            self.edge_indices, self.node_features = self._preprocess_raw_signals(
                metadata_path, data_dir
            )
            self._save_cache()

        self.num_samples = self.node_features.shape[0]
        self.num_nodes = self.node_features.shape[1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.edge_indices[idx], self.node_features[idx]

    def _preprocess_raw_signals(self, metadata_path, data_dir):
        metadata = pd.read_csv(metadata_path)
        adj_list = []
        node_features_list = []
        for id in metadata.subID:
            # Read the raw signals and create correlation matrix
            filepath = os.path.join(data_dir, f"{id}.npy")
            raw_signals = np.load(filepath)
            corr = self._create_corr(raw_signals)

            # Transform correlation matrix to adjacency matrix
            node_features = torch.tensor(corr, dtype=torch.float32)
            topk = node_features.reshape(-1)
            topk, _ = torch.sort(abs(topk), dim=0, descending=True)
            threshold = topk[int(node_features.shape[0] ** 2 / 20 * 2)]
            # TODO: Fix a bug in the line below, where edge_indices might have
            # different shapes because of multiples of the same correlation value
            adj = (torch.abs(node_features) >= threshold).to(int)
            edge_index = dense_to_sparse(adj)[0]

            adj_list.append(edge_index)
            node_features_list.append(node_features)

        return torch.stack(adj_list), torch.stack(node_features_list)

    def _create_corr(self, data):
        eps = 1e-16
        R = np.corrcoef(data)
        R[np.isnan(R)] = 0
        R = R - np.diag(np.diag(R))
        R[R >= 1] = 1 - eps
        corr = 0.5 * np.log((1 + R) / (1 - R))
        return corr

    def _save_cache(self):
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        edge_indices_path = os.path.join(self.cache_path, "edge_indices.pt")
        node_features_path = os.path.join(self.cache_path, "node_features.pt")

        torch.save(self.edge_indices, edge_indices_path)
        torch.save(self.node_features, node_features_path)

    def _cache_exists(self):
        edge_indices_path = os.path.join(self.cache_path, "edge_indices.pt")
        if not os.path.exists(edge_indices_path):
            return False

        node_features_path = os.path.join(self.cache_path, "node_features.pt")
        if not os.path.exists(node_features_path):
            return False

        return True

    def _load_cache(self):
        edge_indices_path = os.path.join(self.cache_path, "edge_indices.pt")
        node_features_path = os.path.join(self.cache_path, "node_features.pt")

        edge_indices = torch.load(edge_indices_path, weights_only=True)
        node_features = torch.load(node_features_path, weights_only=True)
        return edge_indices, node_features
