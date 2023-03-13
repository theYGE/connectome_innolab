# -*- coding: utf-8 -*-
"""
start here
"""
import os
import hydra
import torch
from config import ConnectomeConfig
from omegaconf import OmegaConf
from train import vgae_graph_embedding


@hydra.main(config_path="../conf", config_name="config", version_base="1.3.1")
def main(cfg: ConnectomeConfig) -> None:
    """
    main routine for our software
    """
    # set path to folders
    root = os.path.dirname(os.path.dirname(os.getcwd()))
    assets = os.path.join(root, "assets")
    matrices_path = os.path.join(root, "data", "fc_pt")
    checkpoint_dir = os.path.join(assets, "checkpoints")
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # get configurations
    conf = OmegaConf.to_object(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgae_graph_embedding(conf, device, matrices_path, assets)


if __name__ == "__main__":
    main()
