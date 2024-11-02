import torch
import os.path as osp
import pandas as pd
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, to_undirected
import numpy as np


def get_raw_text_reddit(use_text=False, seed=0):
    if osp.exists(f"./processed_data/reddit.pt"):
        data = torch.load(f"./processed_data/reddit.pt", map_location='cpu')
        edge_index = to_undirected(data.edge_index)
        data.edge_index = edge_index
        data.num_nodes = data.y.shape[0]
        raw_texts = [] # we do not need raw texts for source data, because we already transform them into node features use miniLM
        return data, raw_texts
    else:
        raise NotImplementedError('No existing reddit dataset!')