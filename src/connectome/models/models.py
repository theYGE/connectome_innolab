# -*- coding: utf-8 -*-
"""collection of pytorch or pytorch_geometric NN-models"""
from typing import Tuple
import torch
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn.models.autoencoder import ARGVA
import torch.nn.functional as F


class VarationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VarationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class GCNEncoder1(torch.nn.Module):
    """Implements Graph Convolutional Encoder using GraphSAGE as hidden layers"""

    def __init__(self, in_channels, out_channels, hidden_channels=5) -> None:
        """
        Parameters
        ----------
        in_channels: int number of input channels (data.num_features)
        out_channels: int embedding space
        hidden_channels: int number hidde channels
        """
        super().__init__()
        self.hiddenlayer1 = GINConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logvar = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index) -> Tuple[float, float]:
        """
        Parameters
        ----------
        x: feature(s)
        edge_index: int edge index

        Returns: Tuple of (mu, log_sigma)
        -------
        """
        # x = self.hiddenlayer1(x, edge_index).relu()
        x = self.hiddenlayer1(x, edge_index)
        x = x.relu()
        mu = self.conv_mu(x, edge_index).relu()
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar
