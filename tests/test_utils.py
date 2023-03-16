# -*- coding: utf-8 -*-
"""Do unit tests for functions in module utils."""

import os
import tempfile

import numpy as np
import torch_geometric.nn
import torch_geometric
from src.connectome.utils import utils
from src.connectome.utils import preprocess_matrices
from src.connectome.utils import checks
import torch


def test_graph_data_base():
    """Unittest GraphDataBase"""
    tmp_folder = tempfile.TemporaryDirectory()
    with tmp_folder as tmp:
        random_connectivity_pt = torch.rand(400, 400)
        file_name = os.path.join(tmp, "patient_0")
        torch.save(random_connectivity_pt, file_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        graph = utils.GraphDataBase(root=tmp, device=device)
        assert isinstance(graph, utils.GraphDataBase)


def test_complete_vgae():
    """Unittest complete_vgae"""
    hidden_channels = 3
    dummy_yaml = {
        "vgae": {
            "hidden_out_channels": 5,
            "mu_out_channels": 2,
            "logstd_out_channels": 2,
            "layer_architecture": {
                "_target_": torch_geometric.nn.GraphConv,
                "_partial_": True,
            },
            "activation": {"_target_": torch.nn.ReLU, "_partial_": True},
        }
    }
    vgae = utils.complete_vgae(dummy_yaml, hidden_channels)
    assert vgae.__class__.__name__ == "VariationalGraphAutoEncoder"


def test_set_hidden_layer():
    """Unittest set_hidden_layer"""
    dummy_yaml = {
        "vgae": {
            "layer_architecture": {
                "_target_": torch_geometric.nn.GCNConv,
                "_partial_": True,
            }
        }
    }
    in_channels = 10
    out_channels = 5
    hidden_layer = utils.set_hidden_layer(dummy_yaml, in_channels, out_channels)
    hidden_layer_class_name = hidden_layer.__class__.__name__
    assert hidden_layer_class_name in dir(torch_geometric.nn)


# preprocess_matrices


def test_csv_to_pt():
    """Unittest test_csv_to_pt"""

    tmp_folder_source = tempfile.TemporaryDirectory()
    tmp_folder_destination = tempfile.TemporaryDirectory()

    with tmp_folder_source as source, tmp_folder_destination as destination:
        source_file_name = os.path.join(source, "test_dummy.csv")

        random_csv = torch.rand(400, 400)
        random_csv = random_csv.numpy()
        np.savetxt(source_file_name, random_csv, delimiter=",")
        preprocess_matrices.csv_to_pt(source_path=source, destination_path=destination)
        assert len(os.listdir(destination)) == 1


# checks


def test_check_all_ending():
    """Unittest check_all_ending"""

    tmp_folder = tempfile.TemporaryDirectory()

    with tmp_folder as tmp:
        random_data_csv = torch.rand(400, 400)
        random_data_csv = random_data_csv.numpy()
        np.savetxt(os.path.join(tmp, "dummy_1.csv"), random_data_csv, delimiter=",")
        ramdom_data_npy = torch.rand(400, 400)
        random_data_npy = ramdom_data_npy.numpy()
        np.save(os.path.join(tmp, "dummy_2.npy"), random_data_npy)
        val = checks.check_all_ending(train_data_folder=tmp, ending=".csv")
        assert val == False


if __name__ == "__main__":
    # test_graph_data_base()
    # test_set_hidden_layer()
    # test_complete_vgae()
    test_check_all_ending()
