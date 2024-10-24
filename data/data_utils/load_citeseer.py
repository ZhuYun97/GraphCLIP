from  torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import torch
import os.path as osp
import numpy as np
import random


def get_raw_text_citeseer(use_text=False, seed=0):
    if osp.exists(f"./processed_data/citeseer.pt"):
        data = torch.load(f"./processed_data/citeseer.pt", map_location='cpu')
        data.edge_index = to_undirected(data.edge_index)
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        
        # split data
        node_id = np.arange(data.num_nodes)
        np.random.shuffle(node_id)

        data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
        data.val_id = np.sort(
            node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
        data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])
        

        data.train_mask = torch.tensor(
            [x in data.train_id for x in range(data.num_nodes)])
        data.val_mask = torch.tensor(
            [x in data.val_id for x in range(data.num_nodes)])
        data.test_mask = torch.tensor(
            [x in data.test_id for x in range(data.num_nodes)])

        return data, data.raw_texts
    citeseer_path = './raw_data/CiteSeer-Orig',
    data = Data()
    citeseer_content = osp.join(citeseer_path, "citeseer_texts.txt")
    citeseer_relation = osp.join(citeseer_path, "citeseer.cites")
    idx_to_row_mapping = {}
    category_name = []
    texts = []
    total_num = 0
    l_names = []
    current_l = 0
    l_mapping = {}
    data_y = []
    with open(citeseer_content, "r") as f:
        while True:
            lines = [f.readline().strip() for _ in range(3)]  # Read three lines
            if not any(lines):  # If all lines are empty, end of file reached
                break
            idx_name = lines[0]
            text = lines[1]
            label_name = lines[2]
            texts.append(text)
            idx_to_row_mapping[idx_name] = total_num
            category_name.append(label_name)
            if l_mapping.get(label_name, None) == None:
                l_mapping[label_name] = current_l
                l_names.append(label_name)
                current_l += 1
            data_y.append(l_mapping.get(label_name))
            total_num += 1
    data.y = torch.tensor(data_y)
    row = []
    col = []
    with open(citeseer_relation, "r") as f:
        for line in f.readlines():
            tup = line.split()
            head, tail = tup[0], tup[1]
            head = idx_to_row_mapping.get(head, None)
            tail = idx_to_row_mapping.get(tail, None)
            if head == None or tail == None:
                continue
            if head != tail:
                row.append(head)
                col.append(head)
                row.append(tail)
                col.append(tail)
    data.edge_index = torch.tensor([row, col])    
    data.raw_texts = texts 