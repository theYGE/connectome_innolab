# -*- coding: utf-8 -*-
"""Contains several smaller functions."""
import os
import sys
import re
import pandas as pd
import torch
import torch_geometric
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from config import ConnectomeConfig
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
        Initialize Graph-Database. Each Graph can, but must not have certain labels.

        If labels are given the corresponding csv file must contain a column called 'subject_id'.
        It is important that  connectivity matrices are stored as pytorch tensor(.pt) in directory <root> and file names
        match with corresponding name in subject_id (neglecting file ending .pt in filename).
        For example if root contains files subject_1.pt, subject_2.pt and graph labels are used, there must be rows
        called subject_1 and subject_2 in the subject_id column of file <graph_labels>.

        Args:
            root (str): Absolute path of directory of all adjacency matrices in .pt format.
            device (torch.device): GPU or CPU.
        """
        super().__init__(root)
        assert os.path.exists(root)
        assert isinstance(device, torch.device)
        root_files = os.listdir(root)
        assert len(root_files) >= 1
        root_files = [os.path.join(root, file) for file in root_files]  # abs path here
        root_files.sort()
        self.device = device
        self.root_files = root_files

    def len(self) -> int:
        """
        Calculate length of GraphDataBase object

        Returns:
            int: length of GraphDataBAse object
        """
        return len(self.root_files)

    def get(self, idx: int) -> torch_geometric.data:
        """
        Read connectivity .pt file from root as specified by idx.

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


def complete_vgae(conf: ConnectomeConfig, in_channels: int):
    """
    Complete a partial specified VGAE.

    This function "completes" the VariationalGraphAutoEncoder in module models with information provided at runtime
    ('in_channels') and in configuration file (activation). It assumes that except for <in_channels> all information
    necessary for creating hidden layer is written in config file in subsections 'layer_architecture' and 'activation'
    within the upper 'vgae' section.

    Args:
        conf (ConnectomeConfig): hydra style dictionary of config.yaml file.
        in_channels (int): number of input channels.


    Returns:
        model.VariationalGraphAutoEncoder: the completed VGAE from models module.
    """
    assert isinstance(conf, (DictConfig, dict))
    assert "vgae" in conf
    vgae_dict = conf["vgae"]
    assert "hidden_out_channels" in vgae_dict
    assert "mu_out_channels" in vgae_dict
    assert "logstd_out_channels" in vgae_dict
    assert "layer_architecture" in vgae_dict

    hidden_out_channels = vgae_dict["hidden_out_channels"]
    mu_out_channels = vgae_dict["mu_out_channels"]
    logstd_out_channels = vgae_dict["logstd_out_channels"]
    hidden_layer = set_hidden_layer(conf, in_channels, hidden_out_channels)
    mu_layer = set_hidden_layer(
        conf, in_channels=hidden_out_channels, out_channels=mu_out_channels
    )
    logstd_layer = set_hidden_layer(
        conf, in_channels=hidden_out_channels, out_channels=logstd_out_channels
    )
    activation = vgae_dict["activation"]
    vgae = VariationalGraphAutoEncoder(
        hidden_layer=hidden_layer,
        mu_layer=mu_layer,
        logstd_layer=logstd_layer,
        activation=activation,
    )
    return vgae


def set_hidden_layer(conf: ConnectomeConfig, in_channels: int, out_channels: int):
    """
    Create hidden layer for VGAE.

    Except for number of input_channels which are typically only known
    at runtime all other parameters must be set in config file in 'vgae' entry. It's expected that 'vgae' entry contains
    'layer_architecture' entry which specifies the (partial) hidden NN architecture of the used VGAE.

    Args:
        conf (ConnectomeConfig): hydra style dictionary of config.yaml file.
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.

    Return:
        torch_geometric.nn.<model>
    """
    assert isinstance(conf, (DictConfig, dict))
    assert "vgae" in conf, "Expected to find 'vgae' entry in configuration dictionary."
    assert (
        "layer_architecture" in conf["vgae"]
    ), "Expected to find 'layer_architecture' entry in vgae entry."
    vgae_dict = conf["vgae"]
    layer_partial = instantiate(vgae_dict["layer_architecture"])
    layer_full = layer_partial(in_channels=in_channels, out_channels=out_channels)
    return layer_full


def select_optimal_model(checkpoints_folder: str, log_file: str):
    """
    Select optimal (trained-) Model.

    Based on data in <log_file> pinpoint optimal model in <checkpoints_folder>.
    It is assumed that <log_file> is a csv file containing columns 'epoch', 'train_error', 'validation_error'.

    Args:
        checkpoints_folder (str): Absolute path to folder with fitted model of each epoch.
        log_file (str): Absolute path to file with validation and training error.

    Return:
        torch_geometric.nn.<model>
    """
    assert os.path.isfile(log_file)
    assert os.path.splitext(log_file)[1] == ".csv"
    models = os.listdir(checkpoints_folder)
    models = [os.path.join(checkpoints_folder, filename) for filename in models]
    data_log = pd.read_csv(log_file)
    val_error_series = data_log["validation_error"]
    idx_min = pd.Series.idxmin(val_error_series)
    epoch = data_log["epoch"].iloc[idx_min]
    pattern = "epoch" + str(epoch) + "$"
    to_find = re.compile(pattern)
    optimal_model_list = list(filter(to_find.search, models))
    model = torch.load(optimal_model_list[0])
    return model


if __name__ == "__main__":
    SOURCE_DIST_ROOT = os.path.dirname(os.getcwd())
    SOURCE_DIST_ROOT = os.path.dirname(SOURCE_DIST_ROOT)
    PROJECT_ROOT = os.path.dirname(SOURCE_DIST_ROOT)
    DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
    ASSETS_FOLDER = os.path.join(PROJECT_ROOT, "src", "assets")
    CHECKPOINTS_FOLDER = os.path.join(ASSETS_FOLDER, "checkpoints")
    TRAINING_RESULTS_FOLDER = os.path.join(
        ASSETS_FOLDER, "training_results", "training_03-16-2023-17-19-25_result.csv"
    )
    optimal_model = select_optimal_model(CHECKPOINTS_FOLDER, TRAINING_RESULTS_FOLDER)
