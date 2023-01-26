"""
sets dataclasses of objects used with hydra to support static typing.
"""
from dataclasses import dataclass


@dataclass
class Paths:
    """
    Paths
    """
    graph_adjacencies: str
    graph_labels: str


@dataclass
class Params:
    """
    Parameters
    """
    epochs: int
    out_channel: int
    batch_size: int


@dataclass
class ConnectomeConfig:
    """"
    configuration object
    """
    paths: Paths
    params: Params
