from dataclasses import dataclass


@dataclass
class Paths:
    graph_adjacencies: str
    graph_labels: str


@dataclass
class Params:
    epochs: int
    out_channel: int
    batch_size: int


@dataclass
class ConnectomeConfig:
    paths: Paths
    params: Params
