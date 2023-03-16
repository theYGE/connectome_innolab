# -*- coding: utf-8 -*-
"""Do unit tests for classes in module models."""
from src.connectome.models import models
from src.connectome.utils import utils
import torch_geometric
import torch


def test_variational_graph_auto_encoder():
    """Unittest VariationalAutoEnoderClass"""
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
    vgae = utils.complete_vgae(dummy_yaml, 3)
    vgae_class = vgae.__class__.__name__
    assert vgae_class == models.VariationalGraphAutoEncoder.__name__


if __name__ == "__main__":
    test_variational_graph_auto_encoder()
