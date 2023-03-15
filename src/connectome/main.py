# -*- coding: utf-8 -*-
"""start here"""
import os

import hydra
import torch
from config import ConnectomeConfig
from omegaconf import OmegaConf
from train import vgae_graph_embedding


@hydra.main(config_path="../conf", config_name="config", version_base="1.3.1")
def main(cfg: ConnectomeConfig) -> None:
    """Use this as entry point. Hydra decorator used to read config.yaml file.

    Args:
        cfg (ConnectomeConfig): Hydra dictionary of configuration yaml file.

    """
    # set path to folders
    root = os.path.dirname(os.path.dirname(os.getcwd()))
    assets = os.path.join(root, "src", "assets")
    matrices_path = os.path.join(root, "data", "fc_pt")

    # get configurations
    conf = OmegaConf.to_object(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgae_graph_embedding(conf, device, matrices_path, assets, clear_assets=True)


if __name__ == "__main__":
    main()
