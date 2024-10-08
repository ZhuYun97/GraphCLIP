import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from typing import Any, Dict, Optional
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.nn import GINEConv, GPSConv, global_add_pool, GCNConv, global_mean_pool, SAGEConv, GATConv, GINConv, SAGPooling
from torch_geometric.nn import SimpleConv


non_MP = SimpleConv(aggr='mean', combine_root='sum')


class GPS(torch.nn.Module):
    def __init__(self, in_dim:int, channels: int, out_dim: int, pe_dim: int, num_layers: int,
                 attn_type: str, attn_kwargs: Dict[str, Any]):
        super().__init__()

        self.node_emb = torch.nn.Linear(in_dim, channels - pe_dim)
        self.pe_lin = Linear(32, pe_dim)
        self.pe_norm = BatchNorm1d(32)

        self.convs = ModuleList()
        for l in range(num_layers):

            conv = GPSConv(channels, SAGEConv(channels,channels), heads=8,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels*2, 384),
            )
        self.mlp2 = Sequential(
            Linear(channels, 768),
        )
        self.attn_pool = SAGPooling(channels, 0.1)
        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)
        
        self.lora_A_mlp = Linear(channels*2, 16, bias=False)
        self.lora_B_mlp = Linear(16, 384, bias=False)
        self.lora_A_mlp.weight = torch.nn.Parameter(torch.zeros(16,channels*2))
        self.lora_B_mlp.weight = torch.nn.Parameter(torch.zeros(384, 16))

    def forward(self, x, pe, edge_index, batch, center_idx):
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        for conv in self.convs:
            x = conv(x, edge_index, batch)

        # mean pool
        g_x = global_mean_pool(x, batch)

        c_x = x[center_idx]
        g_x=torch.cat((g_x, c_x), 1) # cat average and center
        
        return self.mlp(g_x), self.mlp2(c_x)


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1
