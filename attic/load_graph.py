import os

# from torch_geometric.data import Data
import networkx as nx
import numpy as np
import torch
import torch_geometric
from matplotlib import pyplot as plt
from torch_geometric.utils import from_networkx

SCRIPT_LOCATION = os.getcwd()  # HOME-DIR/attic
PROJECT_FOLDER = os.path.dirname(SCRIPT_LOCATION)  # HOME-DIR
DATA_FOLDER = os.path.join(PROJECT_FOLDER, "data")

SAMPLE_FILE = os.path.join(DATA_FOLDER, "corr_mat", "rand_corr_1.csv")
SAMPLE_FILE1 = os.path.join(DATA_FOLDER, "corr_mat_pure", "sample.csv")

data1 = np.genfromtxt(SAMPLE_FILE, delimiter=",", names=True, dtype=np.float16)
data2 = np.genfromtxt(SAMPLE_FILE1, delimiter=",", dtype=np.float16)

# mat = np.matrix(data)
# g = nx.read_edgelist(SAMPLE_FILE, create_using=nx.Graph())
# g1 = nx.read_adjlist(SAMPLE_FILE1, create_using=nx.Graph(), delimiter=",")
# G = nx.from_numpy_matrix(data)
print(data1.shape)
print(data2.shape)

g = nx.from_numpy_matrix(np.matrix(data2), create_using=nx.Graph)
g.remove_edges_from(nx.selfloop_edges(g))
print(nx.info(g))
layout = nx.spring_layout(g)

# nx.draw(g, layout)
# nx.draw_networkx_edge_labels(g, pos=layout)
# plt.show()

geom_data = from_networkx(g)
print(type(geom_data))
