import os

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data

SCRIPT_LOCATION = os.getcwd()  # HOME-DIR/attic
PROJECT_FOLDER = os.path.dirname(SCRIPT_LOCATION)  # HOME-DIR
DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")
FCM_FOLDER = os.path.join(DATA_FOLDER, "fc_npy")


root_files = os.listdir(FCM_FOLDER)
idx = 1
adjacency = torch.from_numpy(np.load(os.path.join(FCM_FOLDER, root_files[idx])))
edge_idx, edge_attr = torch_geometric.utils.dense_to_sparse(adjacency)
print(f"edge_idx: {edge_idx}")
print(f"edge_attr: {edge_attr}")
node_features = torch.sum(adjacency, 1)
# print(node_features)
# print(f"node_feature.shape: {node_features.shape}")
# print(f"node_feature.dim(): {node_features.dim()}")
node_features = torch.unsqueeze(node_features, 1)
# print(f"node_feature.shape: {node_features.shape}")
# print(f"node_feature.dim(): {node_features.dim()}")
# print(node_features)
data = Data(x=node_features, edge_index=edge_idx, edge_attribute=edge_attr)
print(data)
