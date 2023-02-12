# -*- coding: utf-8 -*-
"""
several functional utilities.
"""
import hydra
import collections
from omegaconf import OmegaConf
import warnings
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

# TODO: better way?
sys.path.append(os.path.dirname(os.getcwd()))  # /src/connectome/conf/config.yaml
from config import ConnectomeConfig


class GraphDataBase(Dataset):
    """
    Creates a pytorch_geometric Dataset.
    Current use is to load connectome adjacency matrices defined by a 400x400 parcellation of functional connectivity.
    For Schaefer2018_200Parcels_17 atlas see for example
    https://github.com/ThomasYeoLab/CBIG/blob/a8c7a0bb845210424ef1c46d1435fec591b2cf3d/
    stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/
    Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.nii.gz?raw=true
    """

    def __init__(self, root: str, device: torch.device, graph_labels: str = None, set_graph_labels: bool = False):
        """
        Initializes Graph-Database. Each Graph can, but must not have certain labels.
        If labels are given the corresponding csv file must contain a column called 'subject_id'.
        It is important that  connectivity matrices are stored as pytorch tensor(.pt) in directory <root> and file names
        match with corresponding name in subject_id (neglecting file ending .pt in filename).
        For example if root contains files subject_1, subject_2 and graph labels are used, there must be rows called
        subject_1 and subject_2 in the subject_id column of file <graph_labels>.

        Args:
            root: string indicating directory of all adjacency matrices in npy format
            graph_labels: string or list of strings indicating csv file containing graph labels
        """
        super().__init__(root)
        if set_graph_labels:
            assert (graph_labels is None,
                    "cannot have <set_graph_labels> == True and have no valid <graph_labels> filepath")
            warnings.warn('Handling no graph labels needs to be implemented yet')
        if graph_labels is None:
            assert(not set_graph_labels,
                   "cannot have no path to graph labels and <set_graph_labels>==True")
        assert (os.path.exists(root))
        assert (isinstance(device, torch.device))
        root_files = os.listdir(root)
        assert len(root_files) > 1
        root_files = [os.path.join(root, file) for file in root_files]  # abs path here
        root_files.sort()
        self.device = device
        # TODO: functional **and** structural data
        self.root_files = root_files

        if set_graph_labels:
            assert (os.path.isfile(graph_labels))
            _, ending = os.path.splitext(graph_labels)
            assert (ending == ".csv")
        self.graph_labels = graph_labels  # none by default

    def len(self) -> int:
        """
        Returns: int length of dataset
        """
        return len(self.root_files)

    def get(self, idx: int) -> Data:
        """
        Args:
            idx: int index of fetched data

        Returns: torch.data.Dataset
        """
        # edge weights
        file = self.root_files[idx]  # abs path
        adjacency = torch.load(file, map_location=self.device)
        # node feature (=rowSum)
        node_features = torch.sum(adjacency, 1)
        node_features = torch.unsqueeze(node_features, 1)
        # graph feature
        filename = os.path.basename(file)
        filename, _ = os.path.splitext(filename)
        data_id, _ = os.path.splitext(filename)
        edge_idx, edge_attr = torch_geometric.utils.dense_to_sparse(adjacency)
        if self.graph_labels is not None:
            return Data(x=node_features, edge_index=edge_idx, edge_attr=edge_attr)
            graph_labels = pd.read_csv(self.graph_labels)
            graph_labels = graph_labels.loc[graph_labels["subject_id"] == filename]
            graph_labels.drop("subject_id", axis=1, inplace=True)
            graph_labels = graph_labels.to_numpy()
            # TODO:
            # cannot assign Dataset(....., y=graph_labels)
            # return Data with grap label
        else:
            return Data(x=node_features, edge_index=edge_idx, edge_attr=edge_attr)


def show_data() -> None:
    """
    Parameters
    ----------
    cfg: ConnectomeConfig (see config.py)

    Returns: None
    -------
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    graph_labels = os.path.join(root, "data", "covariates", "covariates.csv")
    root = os.path.join(root, "data", "fc_pt")
    # dataset = GraphDataBase(root=root, device=device, graph_labels=graph_labels, set_graph_labels=True).shuffle()
    dataset = GraphDataBase(root=root, device=device)
    print(f"dataset: {dataset}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"loader: {loader}")
    data = dataset[0]
    print(f"data: {data}")


if __name__ == "__main__":
    show_data()
