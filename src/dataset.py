import nibabel as nib
import numpy as np
import os
import pandas as pd
import torch

from torch_geometric.utils import dense_to_sparse
from torch.utils.data import Dataset


NP_TO_TORCH_DTYPES = {
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
}


# Dataset for fMRI data that stores everything in memory
class InMemoryRESTfMRIDataset(Dataset):
    def __init__(
        self,
        metadata_path="./REST-meta-MDD/metadata.csv",
        data_dir="./REST-meta-MDD/fMRI/AAL",
    ):
        super(InMemoryRESTfMRIDataset, self).__init__()
        self.cache_path = os.path.join("cache", data_dir)
        metadata = pd.read_csv(metadata_path)

        if self._cache_exists():
            self.edge_indices, self.node_features = self._load_cache()
        else:
            self.edge_indices, self.node_features = self._preprocess_raw_signals(
                metadata, data_dir
            )
            self._save_cache()

        self.num_samples = self.node_features.shape[0]
        self.num_nodes = self.node_features.shape[1]
        self.labels = torch.tensor(metadata.label.values)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.edge_indices[idx], self.node_features[idx], self.labels[idx]

    def _preprocess_raw_signals(self, metadata, data_dir):
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


class RESTsMRIDataset(Dataset):
    ALLOWED_SPLITS = [
        "c1",  # Gray matter density in native space
        "c2",  # White matter density in native space
        "c3",  # Cerebrospinal fluid density in native space
        "wc1",  # Gray matter density in MNI space
        "wc2",  # White matter density in MNI space
        "wc3",  # Cerebrospinal fluid density in MNI space
        "mwc1",  # Gray matter volume in MNI space
        "mwc2",  # White matter volume in MNI space
        "mwc3",  # Cerebrospinal fluid volume density in MNI space
    ]

    def __init__(
        self,
        data_dir="./REST-meta-MDD/REST-meta-MDD-VBM-Phase1-Sharing",
        splits=["wc1"],  # Can be any combination of allowed splits (["wc1", "mwc1"...])
        normalize=True,  # Normalize images
        dtype=np.float32,  # Reduce to save memory
    ):
        super(RESTsMRIDataset, self).__init__()
        splits = sorted(splits)
        self.cache_path = os.path.join("cache", data_dir, "-".join(splits))
        self.normalize = normalize
        self.dtype = dtype

        for split in splits:
            assert split in self.ALLOWED_SPLITS, f"Split {split} not allowed"


# It's faster but requires more memory. float32 precision takes around 13gb of memory space
class InMemoryRESTsMRIDataset(RESTsMRIDataset):
    def __init__(
        self,
        metadata_path="./REST-meta-MDD/metadata.csv",
        data_dir="./REST-meta-MDD/REST-meta-MDD-VBM-Phase1-Sharing",
        splits=["wc1"],  # Can be any combination of allowed splits (["wc1", "mwc1"...])
        normalize=True,  # Normalize images
        dtype=np.float32,  # Reduce to np.float16 to save memory
    ):
        super(InMemoryRESTsMRIDataset, self).__init__(
            data_dir, splits, normalize, dtype
        )
        metadata = pd.read_csv(metadata_path)

        if self._cache_exists():
            self.data = self._load_cache()
        else:
            self.data = self._preprocess_images(metadata, data_dir, splits)
            self._save_cache()

        self.num_samples = self.data.shape[0]
        self.labels = torch.tensor(metadata.label.values)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def _preprocess_images(self, metadata, data_dir, splits):
        data = np.zeros(
            self._get_data_shape(metadata, data_dir, splits), dtype=self.dtype
        )

        for i, id in enumerate(metadata.subID):
            for j, split in enumerate(splits):
                filepath = os.path.join(data_dir, split, f"{id}.nii.gz")
                image = nib.load(filepath).get_fdata()

                if self.normalize:
                    image -= image.mean()

                data[i][j] = image

        return torch.tensor(data)

    def _save_cache(self):
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        data_path = os.path.join(self.cache_path, "data.pt")
        torch.save(self.data, data_path)

    def _cache_exists(self):
        return os.path.exists(os.path.join(self.cache_path, "data.pt"))

    def _load_cache(self):
        data_path = os.path.join(self.cache_path, "data.pt")
        return torch.load(data_path, weights_only=True)

    def _get_data_shape(self, metadata, data_dir, splits):
        id = metadata.subID[0]
        split = splits[0]
        filepath = os.path.join(data_dir, split, f"{id}.nii.gz")
        image_shape = nib.load(filepath).get_fdata().shape
        return (len(metadata), len(splits), *image_shape)


# It's slower but requires much less memory
class LazyRESTsMRIDataset(RESTsMRIDataset):
    def __init__(
        self,
        metadata_path="./REST-meta-MDD/metadata.csv",
        data_dir="./REST-meta-MDD/REST-meta-MDD-VBM-Phase1-Sharing",
        splits=["wc1"],  # Can be any combination of allowed splits (["wc1", "mwc1"...])
        normalize=True,  # Normalize images
        dtype=np.float32,  # Reduce to save memory
    ):
        super(LazyRESTsMRIDataset, self).__init__(data_dir, splits, normalize, dtype)
        metadata = pd.read_csv(metadata_path)
        self.ids = metadata.subID
        self.labels = torch.tensor(metadata.label.values)

        if not self._cache_exists():
            self._create_cache(metadata, data_dir, splits)

        self.num_samples = len(self.ids)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        filename = self.ids[idx]
        data = np.load(os.path.join(self.cache_path, f"{filename}.npy"))
        return (
            torch.tensor(data, dtype=NP_TO_TORCH_DTYPES[self.dtype]),
            self.labels[idx],
        )

    def _create_cache(self, metadata, data_dir, splits):
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        dshape = self._get_data_shape(metadata, data_dir, splits)[1:]

        for id in self.ids:
            dpoint = np.zeros(dshape, dtype=self.dtype)

            for i, split in enumerate(splits):
                filepath = os.path.join(data_dir, split, f"{id}.nii.gz")
                image = nib.load(filepath).get_fdata()

                if self.normalize:
                    image -= image.mean()

                dpoint[i] = image

            np.save(os.path.join(self.cache_path, f"{id}.npy"), dpoint)

    def _cache_exists(self):
        if not os.path.exists(self.cache_path):
            return False

        for id in self.ids:
            if not os.path.exists(os.path.join(self.cache_path, f"{id}.npy")):
                return False

        return True

    def _get_data_shape(self, metadata, data_dir, splits):
        id = metadata.subID[0]
        split = splits[0]
        filepath = os.path.join(data_dir, split, f"{id}.nii.gz")
        image_shape = nib.load(filepath).get_fdata().shape
        return (len(metadata), len(splits), *image_shape)
