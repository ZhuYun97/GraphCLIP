from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import BatchNorm1d, Identity


def get_activation(name: str):
        activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }
        return activations[name]

class SAGE_Encoder(torch.nn.Module):
    def __init__(self, input_dim, layer_num=2, hidden=128, activation="relu", dropout=0.5, use_bn=True, last_activation=True):
        super(SAGE_Encoder, self).__init__()
        self.layer_num = layer_num
        self.hidden = hidden
        self.input_dim = input_dim
        self.activation = get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)
        self.last_act = last_activation
        self.use_bn = use_bn

        self.convs = ModuleList()
        self.bns = ModuleList()
        # self.acts = ModuleList()
        if self.layer_num > 1:
            self.convs.append(SAGEConv(input_dim, hidden)) 
            for i in range(layer_num-2):
                self.convs.append(SAGEConv(hidden, hidden))
                # glorot(self.convs[i].weight) # initialization
            self.convs.append(SAGEConv(hidden, hidden))
            # glorot(self.convs[-1].weight)
            for i in range(layer_num):
                if use_bn:
                    self.bns.append(BatchNorm1d(hidden))
                else:
                    self.bns.append(Identity())

        else: # one layer gcn
            self.convs.append(SAGEConv(input_dim, hidden)) 
            # glorot(self.convs[-1].weight)
            if use_bn:
                self.bns.append(BatchNorm1d(hidden))
            else:
                self.bns.append(Identity())
            # self.acts.append(self.activation) 
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # print('Inside Model:  num graphs: {}, device: {}'.format(
        #     data.num_graphs, data.batch.device))
        # x, edge_index = data.x, data.edge_index
        for i in range(self.layer_num):
            # x = self.convs[i](x, edge_index, edge_weight)
            # print(i, x.dtype, self.convs[i].lin.weight.dtype)
            x = self.bns[i](self.convs[i](x, edge_index))
            if i == self.layer_num - 1 and not self.last_act:
                pass
                # print(i, 'pass last relu')
            else:
                x = self.activation(x)
            x = self.dropout(x)
            # x = self.activation(self.convs[i](x, edge_index, edge_weight))
            # x = self.bns[i](x)
            # x = self.activation(self.bns[i](self.convs[i](x, edge_index)))
        x = global_mean_pool(x, batch)
        return x
    
    def link(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.layer_num):
            x = self.bns[i](self.convs[i](x, edge_index))
            if i == self.layer_num - 1 and not self.last_act:
                pass
            else:
                x = self.activation(x)
            x = self.dropout(x)
        return x
    
    def reset_parameters(self):
        for i in range(self.layer_num):
            self.convs[i].reset_parameters()
            if self.use_bn:
                self.bns[i].reset_parameters()