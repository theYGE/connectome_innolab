# -*- coding: utf-8 -*-
"""train graph embedding on ceonnectivoty matrices."""
import os.path
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import VGAE
from utils.utils import GraphDataBase
from utils.utils import complete_vgae
from config import ConnectomeConfig
from hydra.utils import instantiate


def vgae_graph_embedding(
    conf: ConnectomeConfig, device, pt_files: str, assets_folder: str
) -> None:
    """
    train graph embedding

    TODO: logger

    Args:
        conf (ConnectomeConfig): Dictionary of configuration file
        device(torch.device): GPU or CPU
        pt_files (str): Path to folder with .pt files
        assets_folder (str): Path to assets folder.
    """
    assert os.path.isdir(assets_folder)
    assert os.path.isdir(pt_files)
    assert isinstance(conf, dict)

    # set params
    checkpoint_dir = os.path.join(assets_folder, "checkpoints")
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    train_share = conf.get("params").get("train_share")
    assert isinstance(train_share, float)
    batch_size = conf.get("params").get("batch_size")
    assert isinstance(batch_size, int)
    optimizer_partial = instantiate(conf.get("params").get("optimizer"))
    epochs = conf.get("params").get("epochs")
    assert isinstance(epochs, int)

    # get data
    dataset = GraphDataBase(root=pt_files, device=device)
    dataset.shuffle()
    train_share = int(len(dataset) * train_share)
    train_dataset = dataset[:train_share]
    val_dataset = dataset[train_share:]
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    num_features = dataset.num_features

    # set up model from partial model with appropriate parameters for completion
    model = VGAE(complete_vgae(conf, num_features))
    # model = VGAE(VarationalGCNEncoder(num_features, 10))
    model.double()
    model.to(device)

    # complete partial optimizer
    optimizer = optimizer_partial(model.parameters())

    for epoch in range(1, epochs + 1):
        print(f"epoch:{str(epoch)}")
        training_loss = 0.0
        validation_loss = 0.0
        model.train()
        for batch_idx, batch in enumerate(train_loader):
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
        print(f"epoch {epoch} training_loss: {training_loss}")

        for batch_idx, batch in enumerate(val_loader):
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
        print(f"epoch {epoch} validation_loss: {validation_loss}")

        model_name = "graph_embedding1_epoch_" + str(epoch)
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        torch.save(model, checkpoint_path)
        print(f"saved model to {checkpoint_path}")
    # # end of epoch


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.getcwd()))
