import torch


class GDataset(torch.utils.data.Dataset):
    def __init__(self, graph_list):
        self.graph_list = graph_list
        
    def __getitem__(self, idx):
        return self.graph_list[idx]
    
    def __len__(self):
        return len(self.graph_list)