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

    For training a folder with connectivity matrices stored as pytorch tensors (.pt files) is needed.
    The module 'utils.preprocess_matrices' provides functionalities for creating a folder with .pt files from a folder
    with .csv files only. Checks on these folders can be performed using functionalities from 'utils.checks.py'.
    Training results will be written in a dedicated assets-folder. During training a folder called 'checkpoints' will
    be created storing models from each epoch as a pickle file (the common format used in 'torch.save()').
    Further a folder called 'training_results' will be created which will store a csv file containing training and
    validation error for each epoch. A hydra folder inside the assets folder stores yaml configurations as well as
    tracked training results from hydra invoked by the decorator of this function.
    The 'checkpoints' and 'training_results' folder may be cleaned before training setting 'clear-assets' to 'True'.
    After training based on training and validation error the model from epoch with lowest validation error can
    loaded using the function 'select_optimal_model' from utils module.



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
