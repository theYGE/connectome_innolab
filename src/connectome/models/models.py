"""
pytorch or pytorch_geometric NN-models
"""
from typing import Tuple
import torch
from torch.nn import Linear
from torch.nn import ReLU
from torch_geometric.nn import GCNConv, SAGEConv


class GCNEncoderSAGE(torch.nn.Module):
    """
    Implements Graph Convolutional Encoder using GraphSAGE as hidden layers
    """
    def __init__(self, in_channels, out_channels, hidden_channels=20) -> None:
        """
        Parameters
        ----------
        in_channels: int number of input channels (data.num_features)
        out_channels: int embedding space
        """
        super().__init__()
        self.hiddenlayer1 = SAGEConv(in_channels, 2 * hidden_channels)
        self.hiddenlayer2 = SAGEConv(2 * hidden_channels, hidden_channels)
        self.mu = SAGEConv(hidden_channels, out_channels)
        self.logvar = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index) -> Tuple[float, float]:
        """
        Parameters
        ----------
        x: feature(s)
        edge_index: int edge index

        Returns: Tuple of (mu, log_sigma)
        -------
        """
        x = self.hiddenlayer1(x, edge_index).relu()
        x = self.hiddenlayer2(x, edge_index).relu()
        mu = self.mu(x, edge_index)
        logvar = self.logvar(x, edge_index)
        return mu, logvar