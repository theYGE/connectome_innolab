# -*- coding: utf-8 -*-
"""train graph embedding on connectivity matrices."""
import os.path

import torch
from config import ConnectomeConfig
from hydra.utils import instantiate
from torch_geometric.loader import DataLoader
from torch_geometric.nn import VGAE
from utils.utils import GraphDataBase, complete_vgae
import pandas as pd
import numpy as np
import shutil
from datetime import datetime


def vgae_graph_embedding(
    conf: ConnectomeConfig,
    device: torch.device,
    pt_files: str,
    assets_folder: str,
    name_prefix: str = "",
    clear_assets: bool = False,
) -> None:
    """
    Train graph variational graph autoencoder.

    Using functional connectivity data in <pt_files> this function trains a graph variational autoencoder.
    Certain training parameters can be set in configuration yaml file. Epoch models created during training will be
    written in a 'checkpoints' folder in <assets_folder> directory. Test and validation data will be written in
    <assets_folder>.

    Args:
        conf (ConnectomeConfig): Dictionary of configuration file
        device (torch.device): GPU or CPU
        pt_files (str): Path to folder with .pt files
        assets_folder (str): Path to assets folder.
        name_prefix (str): If special name for files in assets should be used.
        clear_assets (bool): If True remove 'checkpoint' and 'training_result' folder in assets.
    """
    # TODO: logger
    assert os.path.isdir(assets_folder)
    assert os.path.isdir(pt_files)
    assert isinstance(conf, dict)

    # set params
    checkpoint_dir = os.path.join(assets_folder, "checkpoints")
    results_dir = os.path.join(assets_folder, "training_results")
    time = datetime.now()
    now = str(time.strftime("%m-%d-%Y-%H-%M-%S"))

    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    if clear_assets:
        if os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            os.mkdir(checkpoint_dir)
        if os.path.isdir(results_dir):
            shutil.rmtree(results_dir)
            os.mkdir(results_dir)
    params_conf = conf["params"]
    train_share = params_conf["train_share"]
    assert isinstance(train_share, float)
    batch_size = params_conf["batch_size"]
    assert isinstance(batch_size, int)
    optimizer_partial = instantiate(params_conf["optimizer"])
    epochs = params_conf["epochs"]
    assert isinstance(epochs, int)

    # get data
    dataset = GraphDataBase(root=pt_files, device=device)
    dataset.shuffle()
    train_share = int(len(dataset) * train_share)
    train_dataset = dataset[:train_share]
    val_dataset = dataset[train_share:]
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # num_features = dataset.num_features
    num_features = 1

    # set up model from partial model with appropriate parameters for completion
    model = VGAE(complete_vgae(conf, num_features))
    model.double()
    model.to(device)

    # complete partial optimizer
    optimizer = optimizer_partial(model.parameters())

    val_ts = []
    train_ts = []

    for epoch in range(1, epochs + 1):
        # print(f"epoch:{str(epoch)}")
        training_loss = 0.0
        validation_loss = 0.0
        model.train()
        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x = batch.x
            x.to(device)
            # no split in neg, pos edges are done here
            # therefore relevant edge index is edge index of whole graph
            edge_idx = batch.edge_index

            z = model.encode(x, edge_idx)
            loss = model.recon_loss(z, edge_idx)
            loss = loss + (1 / batch.num_nodes) * model.kl_loss()
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * batch_size
        training_loss /= len(train_loader)
        train_ts.append(training_loss)
        # print(f"epoch {epoch} training_loss: {training_loss}")

        for _, batch in enumerate(val_loader):
            model.eval()
            with torch.no_grad():
                x = batch.x
                x.to(device)
                edge_idx = batch.edge_index
                z = model.encode(x, edge_idx)
                loss = model.recon_loss(z, edge_idx)
                loss = loss + (1 / batch.num_nodes) * model.kl_loss()
                validation_loss += loss.data.item() * batch_size
        validation_loss /= len(val_loader)
        val_ts.append(validation_loss)
        # print(f"epoch {epoch} validation_loss: {validation_loss}")

        if name_prefix == "":
            model_name = "graph_embedding_" + now + "_epoch" + str(epoch)
        else:
            model_name = name_prefix + "_epoch_" + str(epoch)
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        torch.save(model, checkpoint_path)
        # print(f"saved model to {checkpoint_path}")
    # end of epoch
    train_ts = np.array(train_ts)
    train_ts = pd.Series(train_ts).to_frame(name="training_error")
    val_ts = np.array(val_ts)
    val_ts = pd.Series(val_ts).to_frame(name="validation_error")
    epochs = pd.Series(range(1, len(val_ts))).to_frame(name="epoch")
    training_results = pd.concat([epochs, train_ts, val_ts], ignore_index=True)
    if name_prefix == "":
        results_name = "training_" + now + "result_.csv"
    else:
        results_name = name_prefix + ".csv"
    training_results.to_csv(os.path.join(results_dir, results_name))


if __name__ == "__main__":
    pass
