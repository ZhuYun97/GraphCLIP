import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_undirected, remove_isolated_nodes, dropout_adj, remove_self_loops, k_hop_subgraph, to_edge_index, to_dgl
from torch_geometric.utils.num_nodes import maybe_num_nodes
import copy
from torch_sparse import SparseTensor
import dgl


def pyg_random_walk(seeds, graph, length, restart_prob=0.8):
    edge_index = graph.edge_index
    node_num = graph.y.shape[0]
    start_nodes = seeds
    graph_num = start_nodes.shape[0]

    value = torch.arange(edge_index.size(1))

    if type(edge_index) == SparseTensor:
        adj_t = edge_index
    else:
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                    value=value,
                                    sparse_sizes=(node_num, node_num)).t()
        
    current_nodes = start_nodes.clone()

    history = start_nodes.clone().unsqueeze(0)
    signs = torch.ones(graph_num, dtype=torch.bool).unsqueeze(0)
    for i in range(length):
        seed = torch.rand([graph_num])
        nei = adj_t.sample(1, current_nodes).squeeze()
        sign = seed < restart_prob
        nei[sign] = start_nodes[sign]
        history = torch.cat((history, nei.unsqueeze(0)), dim=0)
        signs = torch.cat((signs, sign.unsqueeze(0)), dim=0)
        current_nodes = nei
    history = history.T
    signs = signs.T

    node_list = []
    edge_list = []
    for i in range(graph_num):
        path = history[i]
        sign = signs[i]
        node_idx = path.unique()
        node_list.append(node_idx)

        sources = path[:-1].numpy().tolist()
        targets = path[1:].numpy().tolist()
        sub_edges = torch.IntTensor([sources, targets]).long()
        sub_edges = sub_edges.T[~sign[1:]].T
        # undirectional
        if sub_edges.shape[1] != 0:
            sub_edges = to_undirected(sub_edges)
        edge_list.append(sub_edges)
    return node_list, edge_list


def RWR_sampler(selected_ids, graph, walk_steps=256, restart_ratio=0.5):
    graph  = copy.deepcopy(graph) # modified on the copy
    edge_index = graph.edge_index
    node_num = graph.x.shape[0]
    start_nodes = selected_ids # only sampling selected nodes as subgraphs
    graph_num = start_nodes.shape[0]
    
    value = torch.arange(edge_index.size(1))

    if type(edge_index) == SparseTensor:
        adj_t = edge_index
    else:
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                    value=value,
                                    sparse_sizes=(node_num, node_num)).t()
        
    current_nodes = start_nodes.clone()
    history = start_nodes.clone().unsqueeze(0)
    signs = torch.ones(graph_num, dtype=torch.bool).unsqueeze(0)
    for i in range(walk_steps):
        seed = torch.rand([graph_num])
        nei = adj_t.sample(1, current_nodes).squeeze()
        sign = seed < restart_ratio
        nei[sign] = start_nodes[sign]
        history = torch.cat((history, nei.unsqueeze(0)), dim=0)
        signs = torch.cat((signs, sign.unsqueeze(0)), dim=0)
        current_nodes = nei
    history = history.T
    signs = signs.T

    graph_list = []
    for i in range(graph_num):
        path = history[i]
        sign = signs[i]
        node_idx = path.unique()
        # place the targe index in the first place
        target_idx = path[0].item()
        pos = torch.where(node_idx==target_idx)[0].item()
        if pos != 0:
            tmp = node_idx[0].item()
            node_idx[0] = target_idx
            node_idx[pos] = tmp
        sources = path[:-1].numpy().tolist()
        targets = path[1:].numpy().tolist()
        sub_edges = torch.IntTensor([sources, targets]).long()
        sub_edges = sub_edges.T[~sign[1:]].T
        # undirectional
        if sub_edges.shape[1] != 0:
            sub_edges = to_undirected(sub_edges)
        view = adjust_idx(sub_edges, node_idx, graph, path[0].item())
        view['center_idx'] = target_idx
        view['neig_idx'] = node_idx
        # variables with 'index' will be automatically increased in data loader
        # view = Data(edge_index=sub_edges, x=graph.x[node_idx], center_index=target_idx, center_idx=target_idx, neig_idx=node_idx, y=graph.y[target_idx])

        graph_list.append(view)
    return graph_list

def add_remaining_selfloop_for_isolated_nodes(edge_index, num_nodes):
    num_nodes = max(maybe_num_nodes(edge_index), num_nodes)
    # only add self-loop on isolated nodes
    # edge_index, _ = remove_self_loops(edge_index)
    loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
    connected_nodes_indices = torch.cat([edge_index[0], edge_index[1]]).unique()
    mask = torch.ones(num_nodes, dtype=torch.bool)
    mask[connected_nodes_indices] = False
    loops_for_isolatd_nodes = loop_index[mask]
    loops_for_isolatd_nodes = loops_for_isolatd_nodes.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loops_for_isolatd_nodes], dim=1)
    return edge_index

    
    
def collect_subgraphs(selected_id, graph, walk_steps=20, restart_ratio=0.5):
    graph  = copy.deepcopy(graph) # modified on the copy
    edge_index = graph.edge_index
    node_num = graph.x.shape[0]
    start_nodes = selected_id # only sampling selected nodes as subgraphs
    graph_num = start_nodes.shape[0]
    
    value = torch.arange(edge_index.size(1))

    if type(edge_index) == SparseTensor:
        adj_t = edge_index
    else:
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                    value=value,
                                    sparse_sizes=(node_num, node_num)).t()
    
    current_nodes = start_nodes.clone()
    history = start_nodes.clone().unsqueeze(0)
    signs = torch.ones(graph_num, dtype=torch.bool).unsqueeze(0)
    for i in range(walk_steps):
        seed = torch.rand([graph_num])
        nei = adj_t.sample(1, current_nodes).squeeze()
        sign = seed < restart_ratio
        nei[sign] = start_nodes[sign]
        history = torch.cat((history, nei.unsqueeze(0)), dim=0)
        signs = torch.cat((signs, sign.unsqueeze(0)), dim=0)
        current_nodes = nei
    history = history.T
    signs = signs.T
    
    graph_list = []
    for i in range(graph_num):
        path = history[i]
        sign = signs[i]
        node_idx = path.unique()
        sources = path[:-1].numpy().tolist()
        targets = path[1:].numpy().tolist()
        sub_edges = torch.IntTensor([sources, targets]).long()
        sub_edges = sub_edges.T[~sign[1:]].T
        # undirectional
        if sub_edges.shape[1] != 0:
            sub_edges = to_undirected(sub_edges)
        view = adjust_idx(sub_edges, node_idx, graph, path[0].item())

        graph_list.append(view)
    return graph_list
        
def adjust_idx(edge_index, node_idx, full_g, center_idx):
    '''re-index the nodes and edge index

    In the subgraphs, some nodes are droppped. We need to change the node index in edge_index in order to corresponds 
    nodes' index to edge index
    '''
    # # put center node in the first place
    # pos = torch.where(node_idx==center_idx)[0].item()
    # if pos != 0:
    #     tmp = node_idx[0]
    #     node_idx[0] = center_idx
    #     node_idx[pos] = tmp
    node_idx_map = {j : i for i, j in enumerate(node_idx.numpy().tolist())}
    sources_idx = list(map(node_idx_map.get, edge_index[0].numpy().tolist()))
    target_idx = list(map(node_idx_map.get, edge_index[1].numpy().tolist()))
    edge_index = torch.IntTensor([sources_idx, target_idx]).long()
    x_view = Data(edge_index=edge_index, x=full_g.x[node_idx], y=full_g.y[center_idx], root_n_index=node_idx_map[center_idx])
    return x_view

def ego_graphs_sampler(node_idx, data, hop=2, sparse=False):
    ego_graphs = []
    if sparse:
        edge_index, _ = to_edge_index(data.edge_index)
    else:
        edge_index  = data.edge_index
    row, col = edge_index
    num_nodes = data.x.shape[0]
    for idx in node_idx.numpy().tolist():
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph([idx], hop, edge_index, relabel_nodes=False)
        # sub_edge_index = to_undirected(sub_edge_index)
        pos = torch.where(idx==subset)[0].item()
        if pos != 0:
            tmp = subset[0].item()
            subset[0] = idx
            subset[pos] = tmp
        sub_x = data.x[subset]
        mapping = row.new_full((num_nodes, ), -1)
        mapping[subset] = torch.arange(subset.size(0), device=row.device)
        sub_edge_index = mapping[sub_edge_index]

        # center_idx = subset[mapping].item() # node idx in the original graph, use idx instead
        g = Data(x=sub_x, edge_index=sub_edge_index, root_n_index=mapping, y=data.y[idx], original_idx=subset) # note: there we use root_n_index to record the index of target node, because `PyG` increments attributes by the number of nodes whenever their attribute names contain the substring :obj:`index`
        g['center_idx'] = idx
        g['neig_idx'] = subset
        ego_graphs.append(g)
    return ego_graphs