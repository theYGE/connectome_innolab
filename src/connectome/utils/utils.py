# -*- coding: utf-8 -*-
"""
This module contains several smaller functions to reduce overall LOC and hence increase readability as well as handling
parameters set in configuration yaml file.
"""

import os
import re
from sys import path as spath
from matplotlib import pyplot as plt
import pandas as pd

import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from hydra.utils import instantiate
from omegaconf import DictConfig
from omegaconf import OmegaConf


# TODO: better way to include modules?
spath.append(os.path.dirname(os.getcwd()))
from config import ConnectomeConfig  # declares classes of config yaml
from models.models import VariationalGraphAutoEncoder


class GraphDataBase(Dataset):
    """
    Creates a pytorch_geometric Dataset.
    Current use is to load connectome adjacency matrices defined by a 400x400 parcellation of functional connectivity.
    For Schaefer2018_200Parcels_17 atlas see for example
    https://github.com/ThomasYeoLab/CBIG/blob/a8c7a0bb845210424ef1c46d1435fec591b2cf3d/
    stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/
    Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.nii.gz?raw=true
    """

    def __init__(self, root: str, device: torch.device):
        """
        Initializes Graph-Database. Each Graph can, but must not have certain labels.
        If labels are given the corresponding csv file must contain a column called 'subject_id'.
        It is important that  connectivity matrices are stored as pytorch tensor(.pt) in directory <root> and file names
        match with corresponding name in subject_id (neglecting file ending .pt in filename).
        For example if root contains files subject_1.pt, subject_2.pt and graph labels are used, there must be rows called
        subject_1 and subject_2 in the subject_id column of file <graph_labels>.

        Args:
            root: string indicating directory of all adjacency matrices in npy format
            graph_labels: string or list of strings indicating csv file containing graph labels
        """
        super().__init__(root)
        assert os.path.exists(root)
        assert isinstance(device, torch.device)
        root_files = os.listdir(root)
        assert len(root_files) > 1
        root_files = [os.path.join(root, file) for file in root_files]  # abs path here
        root_files.sort()
        self.device = device
        self.root_files = root_files

    def len(self) -> int:
        """
        Calculates length of GraphDataBase object

        Returns:
            int: length of GraphDataBAse object
        """
        return len(self.root_files)

    def get(self, idx: int) -> Data:
        """
        Reads a connectivity pt file from root as specified by idx.
        Args:
            idx (int): index of fetched data

        Returns:
            torch.data.Dataset: pyG graph of connectivity pt file.
        """
        file = self.root_files[idx]
        adjacency = torch.load(file, map_location=self.device)
        # create row sum as node feature
        node_features = torch.sum(adjacency, 1)
        node_features = torch.unsqueeze(node_features, 1)
        edge_idx, edge_attr = torch_geometric.utils.dense_to_sparse(adjacency)

        return Data(x=node_features, edge_index=edge_idx, edge_attr=edge_attr)


def show_data() -> None:
    """ """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def plot_training(
    path_csv: str, path_to_store: str = None, filename: str = None, rename_dict=None
) -> None:
    """
    plot training error
    """
    assert os.path.isdir(path_csv)
    # assert (path_to_store is not None & filename is not None) | (
    #    path_to_store is None & filename is None
    # )

    files = os.listdir(path_csv)
    files = [os.path.join(path_csv, file) for file in files]
    df = pd.DataFrame()
    for file in files:
        group_name = os.path.basename(file)
        if group_name in rename_dict:
            group_name = rename_dict.get(group_name)
        else:
            group_name = os.path.splitext(group_name)[0]
        dataframe = pd.read_csv(file)
        dataframe.set_index(["epoch"], inplace=True)
        dataframe["group"] = group_name
        df = pd.concat([df, dataframe])

    df = df.reset_index(level=0)
    dfp = df.pivot(index="epoch", columns="group", values="train_error")
    dfp.plot()
    plt.ylabel("validation error")
    if not path_to_store is None:
        plt.savefig(os.path.join(path_to_store, filename))
    plt.show()


def complete_vgae(conf: ConnectomeConfig, in_channels: int):
    """
    This function "completes" the VariationalGraphAutoEncoder in module models with information provided at runtime
    ('in_channels') and in configuration file (activation). It assumes that except for <in_channels> all information
    necessary for creating hidden layer is written in config file in subsections 'layer_architecture' and 'activation'
    within the upper 'vgae' section.
    TODO: static typing **any** from torch_geometry.nn.<model>
    Args:
        conf (ConnectomeConfig): hydra style dictionary of config.yaml file.
        in_channels (int): number of input channels.
    Returns:
        model.VariationalGraphAutoEncoder: the completed VGAE from models module.
    """
    assert isinstance(conf, (DictConfig, dict))
    assert "vgae" in conf
    assert "hidden_out_channels"
    assert "mu_out_channels"
    assert "logstd_out_channels"
    assert "layer_architecture"
    hidden_out_channels = conf.get("vgae").get("hidden_out_channels")
    mu_out_channels = conf.get("vgae").get("mu_out_channels")
    logstd_out_channels = conf.get("vgae").get("logstd_out_channels")
    hidden_layer = set_hidden_layer(conf, in_channels, hidden_out_channels)
    mu_layer = set_hidden_layer(
        conf, in_channels=hidden_out_channels, out_channels=mu_out_channels
    )
    logstd_layer = set_hidden_layer(
        conf, in_channels=hidden_out_channels, out_channels=logstd_out_channels
    )

    activation = conf.get("vgae").get("activation")
    # mu_layer
    vgae = VariationalGraphAutoEncoder(
        hidden_layer=hidden_layer,
        mu_layer=mu_layer,
        logstd_layer=logstd_layer,
        activation=activation,
    )
    return vgae


def set_hidden_layer(conf: ConnectomeConfig, in_channels: int, out_channels: int):
    """
    This function returns the hidden layer for VGAE. Except for number of input_channels which are typically only known
    at runtime all other parameters must be set in config file in 'vgae' entry. It's expected that 'vgae' entry contains
    'layer_architecture' entry which specifies the (partial) hidden NN architecture of the used VGAE.

    TODO: static type checking **any** from torch_geometry.nn

    Args:
        conf (ConnectomeConfig): hydra style dictionary of config.yaml file.
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.

    Returns:
        torch_geometric.nn.<model>
    """
    assert isinstance(conf, (DictConfig, dict))
    assert "vgae" in conf, "Expected to find 'vgae' entry in configuration dictionary."
    assert "layer_architecture" in conf.get(
        "vgae"
    ), "Expected to find 'layer_architecture' entry in vgae entry."
    layer_partial = instantiate(conf.get("vgae").get("layer_architecture"))
    layer_full = layer_partial(in_channels=in_channels, out_channels=out_channels)
    return layer_full


def select_optimal_model(checkpoints_folder: str, log_file: str, erase: bool = True):
    """
    Based on data in <log_file> pinpoint optimal model in <checkpoints_folder>
    """
    assert os.path.isfile(log_file)
    assert os.path.splitext(log_file)[1] == ".csv"
    models = os.listdir(checkpoints_folder)
    models = [os.path.join(checkpoints_folder, filename) for filename in models]
    data_log = pd.read_csv(log_file)
    val_error_series = data_log["validation_error"]
    idx_min = pd.Series.idxmin(val_error_series)
    epoch = data_log["epoch"].iloc[idx_min]
    pattern = "epoch_" + str(epoch) + "$"
    to_find = re.compile(pattern)
    optimal_model_list = list(filter(to_find.search, models))
    model = torch.load(optimal_model_list[0])
    print(model)


if __name__ == "__main__":
    SOURCE_DIST_ROOT = os.path.dirname(os.getcwd())
    SOURCE_DIST_ROOT = os.path.dirname(SOURCE_DIST_ROOT)
    PROJECT_ROOT = os.path.dirname(SOURCE_DIST_ROOT)
    DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
    ASSETS_FOLDER = os.path.join(PROJECT_ROOT, "assets")
    conf_dictionary = OmegaConf.load(
        "/home/svenmaurice/05_version_control/01_github/connectome_innolab/conf/config.yaml"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patient_fc_path = os.path.join(DATA_FOLDER, "fc_pt")
    model_path = os.path.join(ASSETS_FOLDER, "checkpoints", "graph_embedding1_epoch_1")

    # models_checkpoints = os.path.join(ASSETS_FOLDER, "checkpoints")
    # model_log = os.path.join(ASSETS_FOLDER, "log", "2023-03-08:20:11:53.csv")
    # select_optimal_model(models_checkpoints, model_log, erase=False)
    model = complete_vgae(conf_dictionary, 20)
    print(model)
