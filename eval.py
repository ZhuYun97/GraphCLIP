import argparse
import os.path as osp
import json
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric import seed_everything
from torch_geometric.data import Data
import torch.multiprocessing as mp
import torch.distributed as dist
import os
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.nn import DataParallel
import copy

from data.load import load_data
from models import GraphCLIP, GraphMAE
from models.projector import Projector
from src.transformers.models.bert.tokenization_bert import BertTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.utils import dropout_adj


def split_dataloader(data, graphs, batch_size, seed=0, name='cora'):
    train_idx = data.train_mask.nonzero().squeeze()
    val_idx = data.val_mask.nonzero().squeeze()
    test_idx = data.test_mask.nonzero().squeeze()
    train_dataset = [graphs[idx] for idx in train_idx]
    val_dataset = [graphs[idx] for idx in val_idx]
    test_dataset = [graphs[idx] for idx in test_idx]
    torch.save(train_dataset, f=f"./eval/tiny_mean_train_{name}-{seed}.pt")
    torch.save(val_dataset, f=f"./eval/tiny_mean_val_{name}-{seed}.pt")
    torch.save(test_dataset, f=f"./eval/tiny_mean_test_{name}-{seed}.pt")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # use DataListLoader for DP rather than DataLoader
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def parse_data(name, data):
    transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
    json_data = []
    with open(f'./graphtext/summary-{name}.json', 'r') as fcc_file:
        fcc_data = json.load(fcc_file)
        json_data = fcc_data

    collected_graph_data = []
    # collected_text_data = []
    for id, jd in enumerate(json_data):
        assert id == jd['id']
        edges = torch.tensor(jd['graph'])
        summary = jd['summary']
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

def _drop_feature(g, rate):
    new_g = copy.deepcopy(g)
    drop_mask = torch.empty(
        (g.x.size(1), ),
        dtype=torch.float32,
        device=g.x.device).uniform_(0, 1) < rate
    
    new_g.x[:, drop_mask] = 0
    return new_g

def graph_aug(g, f_p, e_p):
    new_g = copy.deepcopy(g)
    drop_mask = torch.empty(
        (g.x.size(1), ),
        dtype=torch.float32,
        device=g.x.device).uniform_(0, 1) < f_p
    
    new_g.x[:, drop_mask] = 0
    e, _ = dropout_adj(new_g.edge_index, p=e_p)
    new_g.edge_index = e
    return new_g


seed_everything(88) # 88

attn_kwargs = {'dropout': 0.0}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
text_model = 'tiny'
model = GraphCLIP(384, 1024, 12, attn_kwargs, text_model=text_model)
model.load_state_dict(torch.load("./ckpts/graphclip.pt"), strict=False)

model.to(device)

print("mdoel is loaded")


batch_size= 1024 # 800*8



################ target data
target_data = "cora*citeseer*wikics*history*photo*instagram*computer".split("*")
target_datasets = target_data
target_classes_list = []
target_c_desc_list = []
target_test_loaders = []
for d in target_data:
    data, text, classes, c_descs = load_data(d, seed=0)
    target_classes_list.append(classes)
    target_c_desc_list.append(c_descs)
    target_graph = parse_data(d, data)
    _, _, target_test_loader = split_dataloader(data, target_graph, batch_size, seed=0,name=d)
    
    target_test_loaders.append(target_test_loader)
print("seed 0 is loaded")



parser = argparse.ArgumentParser()
parser.add_argument(
    '--attn_type', default='multihead',
    help="Global attention type such as 'multihead' or 'performer'.")
args = parser.parse_args()





tokenizer = BertTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

eval_template={
    'cora': "this paper has a topic on {c}", # "it belongs to {c} research area",
    'citeseer': "good paper of {c} ", # "it belongs to {c} research area",
    'pubmed': "it belongs to {c} research area",
    'arxiv_2023': "it belongs to {c} research area",
    'wikics': "it belongs to {c} research area",
    'photo':  "this product belongs to {c}",
    'computer':  "is {c} category", # "this product belongs to {c}",
    'history': "this book belongs to {c}",
    # 'instagram': "Based on the profile provided, this account is a {c} account on Instagram. ",
    'instagram': "{c}",
    'reddit': "{c}"
}

@torch.no_grad()
def test(loader, classes, c_descs, dataset_name):
    model.eval()

    text_inputs = [eval_template[dataset_name].format(c=c) for c in classes]
    text_inputs = [ti+desc for ti, desc in zip(text_inputs, c_descs)]
    correct = 0

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        batch_t = tokenizer(text_inputs, truncation=True, padding=True, return_tensors="pt", max_length=512).to(device)

        with torch.no_grad():
            graph_embs, _ = model.encode_graph(batch)
            text_embs = model.encode_text(batch_t["input_ids"], batch_t['token_type_ids'], batch_t["attention_mask"])
            graph_embs /= graph_embs.norm(dim=-1, keepdim=True)
            text_embs /= text_embs.norm(dim=-1, keepdim=True)
            similarity = (100.0 * graph_embs @ text_embs.T).softmax(dim=-1)
            y = batch.y
            correct += torch.sum(similarity.argmax(dim=1) == y).item()

    return correct / len(loader.dataset)


res_str = ""
all_test_list = []

run_test = []
for i, classes in enumerate(target_classes_list):
    test_acc = test(target_test_loaders[i], classes, target_c_desc_list[i], target_datasets[i])
    run_test.append(test_acc)
    res_str += f" {target_datasets[i]} acc: {test_acc}"
print(1, res_str)

