import torch
import os.path as osp


def get_raw_text_pubmed(use_text=False, seed=0):
    if osp.exists(f"./processed_data/pubmed.pt"):
        data = torch.load(f"./processed_data/pubmed.pt", map_location='cpu')
        data.num_nodes = data.y.shape[0]
        raw_texts = [] # we do not need raw texts for source data, because we already transform them into node features use miniLM
        return data, raw_texts
    else:
        raise NotImplementedError('No existing pubmed dataset!')