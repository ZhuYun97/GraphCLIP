import torch
import json
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from tqdm import tqdm


def parse_source_data(name, data):
    transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
    json_data = []

    with open(f'./summary/summary-{name}.json', 'r') as fcc_file: # subgraph-summary pair
        fcc_data = json.load(fcc_file)
        json_data = fcc_data

    collected_graph_data = []
    # collected_text_data = []
    print("process", name)
    for id, jd in enumerate(tqdm(json_data)):
        assert id == jd['id']
        edges = torch.tensor(jd['graph'])
        summary = jd['summary']
        # reindex
        node_idx = torch.unique(edges)
        node_idx_map = {j : i for i, j in enumerate(node_idx.numpy().tolist())}
        sources_idx = list(map(node_idx_map.get, edges[0].numpy().tolist()))
        target_idx = list(map(node_idx_map.get, edges[1].numpy().tolist()))
        edge_index = torch.IntTensor([sources_idx, target_idx]).long()
        graph = Data(edge_index=edge_index, x=data.x[node_idx], y=data.y[jd['id']], root_n_index=node_idx_map[jd['id']], summary=summary)
        graph=transform(graph) # add PE
        collected_graph_data.append(graph)
    return collected_graph_data

def parse_target_data(name, data):
    transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
    json_data = []
    with open(f'./target_data/{name}.json', 'r') as fcc_file:
        fcc_data = json.load(fcc_file)
        json_data = fcc_data

    collected_graph_data = []
    # collected_text_data = []
    for id, jd in enumerate(json_data):
        assert id == jd['id']
        edges = torch.tensor(jd['graph'])
        if edges.shape[1] == 0:
            edges = torch.tensor([[id],[id]])
        # summary = jd['summary']
        # reindex
        node_idx = torch.unique(edges)
        node_idx_map = {j : i for i, j in enumerate(node_idx.numpy().tolist())}
        sources_idx = list(map(node_idx_map.get, edges[0].numpy().tolist()))
        target_idx = list(map(node_idx_map.get, edges[1].numpy().tolist()))
        edge_index = torch.IntTensor([sources_idx, target_idx]).long()
        graph = Data(edge_index=edge_index, x=data.x[node_idx], y=data.y[jd['id']], root_n_index=node_idx_map[jd['id']])
        graph=transform(graph) # add PE

        collected_graph_data.append(graph)
        # collected_text_data.append(summary)
    return collected_graph_data

def split_dataloader(data, graphs, batch_size, seed=0, name='cora'):
    train_idx = data.train_mask.nonzero().squeeze()
    val_idx = data.val_mask.nonzero().squeeze()
    test_idx = data.test_mask.nonzero().squeeze()
    train_dataset = [graphs[idx] for idx in train_idx]
    val_dataset = [graphs[idx] for idx in val_idx]
    test_dataset = [graphs[idx] for idx in test_idx]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # use DataListLoader for DP rather than DataLoader
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader