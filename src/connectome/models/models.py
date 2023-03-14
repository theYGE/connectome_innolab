# -*- coding: utf-8 -*-
"""This module contains the model(s) used for training GNN on functional connectivity matrices from UKB"""

import torch


class VariationalGraphAutoEncoder(torch.nn.Module):
    """
    Class for Graph Variational Autoencoder.

    Only the Encoder is specified. PyG uses by default inner product
    Decoder. See
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.VGAE.html#torch_geometric.nn.models.VGAE
    and corresponding paper for justification.
    """

    def __init__(self, hidden_layer, mu_layer, logstd_layer, activation):
        """
        Create instance of a VGAE object.

        Hidden layers can to some extent be set in config yaml file.
        Currently only one hidden layer is supported which is used for mu and log_sigma as well.
        The user may specify a network architecture of his choice provided input and output channels are used next to
        hyperparameters that must be specified in yaml file.

        Args:
            hidden_layer (torch_geometric.nn.<model>): complete NN architecture for hidden layer.
            mu_layer (torch_geometric.nn.<model>): complete NN architecture for mu.
            logstd_layer (torch_geometric.nn.<model>): complete NN architexture for log_std.
            activation (torch.nn.<activation>): activation function to be used
        Returns:
            torch._geometric.nn.VGAE: Graph variational Autoencodder
        """
        super(VariationalGraphAutoEncoder, self).__init__()
        self.hidden_layer = hidden_layer
        self.mu = mu_layer
        self.log_std = logstd_layer

    def forward(self, x, edge_idx):
        """Forward pass of encoder network."""
        x = self.hidden_layer(x, edge_idx).relu()
        mu = self.mu(x, edge_idx)
        log_std = self.log_std(x, edge_idx)
        return mu, log_std
