# -*- coding: utf-8 -*-
""""""
import torch


class VariationalGraphAutoEncoder(torch.nn.Module):
    """
    Class for Graph Variational Autoencoder. Only the Encoder is specified. PyG uses by default inner product
    Decoder. See
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.VGAE.html#torch_geometric.nn.models.VGAE
    and corresponding paper for justification.
    """

    def __init__(self, hidden_layer, mu_layer, logstd_layer, activation):
        """
        Creates instance of a VGAE object. Note that the hidden layer will be typically created using 'utils.set_layer'
        and activation with 'utils.set_activation'.

        Args:
            hidden_layer (torch_geometric.nn.<model>): complete NN architecture for hidden layer.

        Returns:
            torch._geometric.nn.VGAE: Graph variational Autoencodder
        """
        super(VariationalGraphAutoEncoder, self).__init__()
        self.hidden_layer = hidden_layer
        self.mu = mu_layer
        self.log_std = logstd_layer

    def forward(self, x, edge_idx):
        x = self.hidden_layer(x, edge_idx).relu()
        mu = self.mu(x, edge_idx)
        log_std = self.log_std(x, edge_idx)
        return mu, log_std
