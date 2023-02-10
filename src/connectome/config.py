# -*- coding: utf-8 -*-
"""
sets dataclasses of objects used with hydra to support static typing.
"""
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class Params:
    """Parameters"""
    epochs: int
    out_channels: int
    lr: float
    batch_size: int
    num_workers: int
    train_share: float


@dataclass
class ConnectomeConfig:
    """"configuration object"""
    params: Params


cs = ConfigStore.instance()
cs.store(name="base_config", node=ConnectomeConfig)
