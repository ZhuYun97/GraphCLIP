from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
import os.path as osp


data_path = '../datasets'
def get_raw_text_arxiv(use_text=False, seed=0):
    if osp.exists(f"./processed_data/arxiv.pt"):
        data = torch.load(f"./processed_data/arxiv.pt", map_location='cpu')
        data.num_nodes = data.y.shape[0]
        raw_texts = [] # we do not need raw texts for source data, because we already transform them into node features use miniLM
        return data, raw_texts

    dataset = PygNodePropPredDataset(root=data_path,
        name='ogbn-arxiv', transform=T.ToSparseTensor())
    data = dataset[0]

    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.y = data.y.squeeze()

    data.edge_index = data.adj_t.to_symmetric()
    if not use_text:
        return data, None

    nodeidx2paperid = pd.read_csv(
        f'{data_path}/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')

    raw_text = pd.read_csv(f'{data_path}/arxiv_orig/titleabs.tsv',
                           sep='\t', header=None, names=['paper id', 'title', 'abs'], skiprows=[0])
    raw_text = raw_text.dropna()

    # nodeidx2paperid['paper id'] = nodeidx2paperid['paper id'].astype('int64')
    raw_text['paper id'] = raw_text['paper id'].astype('int64')
    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')

    text = []
    for ti, ab in zip(df['title'], df['abs']):
        t = 'Title: ' + ti + '\n' + 'Abstract: ' + ab
        text.append(t)
    return data, text
