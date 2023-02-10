# -*- coding: utf-8 -*-
import os.path
import hydra
import torch
from omegaconf import OmegaConf
from config import ConnectomeConfig
from utils.utils import GraphDataBase
from torch_geometric.loader import DataLoader
from models.models import GCNEncoder1
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import VGAE
from torch.optim import Adam


@hydra.main(config_path="conf", config_name="config", version_base="1.3.1")  # entry point
def graph_embedding1(cfg: ConnectomeConfig):
    """

    """
    root = os.path.dirname(os.path.dirname(os.getcwd()))
    assets = os.path.join(root, "src", "connectome", "assets")
    graph_label = os.path.join(root, "data", "covariates", "covariates.csv")
    matrices_path = os.path.join(root, "data", "fc_pt")

    conf = OmegaConf.to_object(cfg)
    # TODO: check if entry found in dictionary
    batch_size = conf.get("params").get("batch_size")
    train_share = conf.get("params").get("train_share")
    out_channels = conf.get("params").get("out_channels")
    learning_rate = conf.get("params").get("lr")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = GraphDataBase(root=matrices_path, device=device, graph_labels=graph_label)
    dataset.shuffle()
    dataset = dataset[0:300]
    dataloader = DataLoader(dataset, batch_size=batch_size)
    train_share = int(len(dataset) * train_share)
    train_dataset = dataset[:train_share]
    val_dataset = dataset[train_share:]
    # train_loader = DataLoader(train_dataset)
    # val_loader = DataLoader(val_dataset)
    # sample_data = dataset[0]

    num_features = dataset.num_features
    model = VGAE(GCNEncoder1(num_features, out_channels))
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(log_dir=os.path.join(assets, "tensorboard"))

    for epoch in range(1, conf.get("params").get("epochs") + 1):
        print(f"epoch: {epoch}")
        training_loss = 0.0
        validation_loss = 0.0
        model.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):

            x = batch.x
            edge_idx = batch.edge_index
            z = model.encode(x, edge_idx)
            loss = model.recon_loss(z, edge_idx)
            loss = loss + (1 / batch.num_nodes) * model.kl_loss()
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * batch_size
        training_loss /= len(dataloader)
        # TODO: add validation
        # TODO: pretty tensorboard data
        writer.add_scalars(training_loss, epoch)
        torch.save(model.state_dict(), os.path.join(assets, "model"))


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.getcwd()))
    #graph_embedding1()
    x = load_model(os.path.join(root, "assets", "model"))
    print(type(x))
