# -*- coding: utf-8 -*-
import os.path
import hydra
from config import ConnectomeConfig
from utils.utils import GraphDataBase
from torch_geometric.loader import DataLoader
from models.models import *
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import VGAE
from torch.optim import Adam


def graph_embedding1(root: str, assets: str, graph_label: str, matrices_path: str, checkpoint_dir: str,
                     batch_size: int, train_share: float, out_channels: int, device: torch.device,
                     learning_rate: float, epochs: int) -> None:
    """
    train graph embedding
    """
    dataset = GraphDataBase(root=matrices_path, device=device, graph_labels=graph_label)
    dataset.shuffle()
    #dataset = dataset[0:300]
    train_share = int(len(dataset) * train_share)
    train_dataset = dataset[:train_share]
    val_dataset = dataset[train_share:]
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    num_features = dataset.num_features

    model = VGAE(VarationalGCNEncoder(num_features, out_channels))
    model.double()

    optimizer = Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "tensorboard"))

    for epoch in range(1, epochs + 1):
        print(f"epoch:{str(epoch)}")
        training_loss = 0.0
        validation_loss = 0.0
        model.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            x = batch.x
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
        writer.add_scalar("train_loss", training_loss, epoch)
        # TODO: tensorboard data
        for batch_idx, batch in enumerate(val_loader):
            model.eval()
            with torch.no_grad():
                x = batch.x
                edge_idx = batch.edge_index
                z = model.encode(x, edge_idx)
                loss = model.recon_loss(z, edge_idx)
                loss = loss + (1 / batch.num_nodes) * model.kl_loss()
                validation_loss += loss.data.item() * batch_size
        validation_loss /= len(val_loader)
        print(f"epoch {epoch} validation_loss: {validation_loss}")
        writer.add_scalar("val_loss", validation_loss, epoch)

        model_name = "graph_embedding1_epoch_" + str(epoch) + ".pickle"
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"saved model to {checkpoint_path}")


@hydra.main(config_path="conf", config_name="config", version_base="1.3.1")  # entry point
def show_embedding(conf: ConnectomeConfig):
    out_channels = conf.get("params").get("out_channels")

    root = os.path.dirname(os.path.dirname(os.getcwd()))
    graph_label = os.path.join(root, "data", "covariates", "covariates.csv")
    assets = os.path.join(root, "src", "connectome", "assets")
    matrices_path = os.path.join(root, "data", "fc_pt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_data = dataset = GraphDataBase(root=matrices_path, device=device, graph_labels=graph_label)
    num_features = dataset.num_features
    model = VGAE(VarationalGCNEncoder(num_features, out_channels))
    checkpoint = torch.load(os.path.join(root, "assets", "model"))
    print(f"checkpoint: {checkpoint}")
    model.load_state_dict(checkpoint["model"])
    print(f"model: {model}")
    test_data = dataset[:10]
    for data in test_data:
        x = data.x
        edge_idx = data.edge_index
        z = model.encode(x, edge_idx)
        print(z)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.getcwd()))
    graph_embedding1()
    # show_embedding()
