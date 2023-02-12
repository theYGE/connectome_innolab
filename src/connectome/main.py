# -*- coding: utf-8 -*-
"""
start here
"""
import os
import torch
import hydra
from omegaconf import OmegaConf
from config import ConnectomeConfig
from embedding1 import graph_embedding1


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3.1")  # entry point
def main(cfg: ConnectomeConfig) -> None:
    """
    main routine for our software
    """
    # set path to folders
    root = os.path.dirname(os.path.dirname(os.getcwd()))
    assets = os.path.join(root, "assets")
    graph_label = os.path.join(root, "data", "covariates", "covariates.csv")
    matrices_path = os.path.join(root, "data", "fc_pt")
    checkpoint_dir = os.path.join(assets, "checkpoints")
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # get configurations
    conf = OmegaConf.to_object(cfg)
    # TODO: check if entry found in dictionary
    batch_size = conf.get("params").get("batch_size")
    train_share = conf.get("params").get("train_share")
    out_channels = conf.get("params").get("out_channels")
    learning_rate = conf.get("params").get("lr")
    epochs = conf.get("params").get("epochs")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train graph embedding
    graph_embedding1(root=root, assets=assets, graph_label=graph_label, matrices_path=matrices_path,
                     checkpoint_dir=checkpoint_dir, batch_size=batch_size, train_share=train_share,
                     out_channels=out_channels, device=device, learning_rate=learning_rate, epochs=epochs)


if __name__ == "__main__":
    main()
