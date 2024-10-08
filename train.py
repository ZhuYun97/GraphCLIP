import argparse
import json
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric.transforms as T
from torch_geometric.loader import DataListLoader
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.nn import DataParallel
from transformers import AutoTokenizer
import copy

from data.load import load_data
from data.split import split_dataloader
from models import GraphCLIP
from utils.augmentation import adversarial_aug_train
from torch_geometric.utils import dropout_edge



def graph_aug(g, f_p, e_p):
    new_g = copy.deepcopy(g)
    drop_mask = torch.empty(
        (g.x.size(1), ),
        dtype=torch.float32,
        device=g.x.device).uniform_(0, 1) < f_p
    
    new_g.x[:, drop_mask] = 0
    e, _ = dropout_edge(new_g.edge_index, p=e_p)
    new_g.edge_index = e
    return new_g

def parse_data(name, data):
    transform = T.AddRandomWalkPE(walk_length=32, attr_name='pe')
    json_data = []

    with open(f'./graphtext/summary-{name}.json', 'r') as fcc_file: # subgraph-summary pair
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
        graph = Data(edge_index=edge_index, x=data.x[node_idx], y=data.y[jd['id']], root_n_index=node_idx_map[jd['id']], summary=summary)
        graph=transform(graph) # add PE
        collected_graph_data.append(graph)
    return collected_graph_data



seed_everything(88) # 88

source_dataname="ogbn-arxiv"
source_data, source_text, source_classes, source_c_descs = load_data(source_dataname, seed=0)
source_graph = parse_data(source_dataname, source_data)

source_data2, source_text2, source_classes2, source_c_descs2 = load_data('reddit', seed=0)
source_graph2 = parse_data('reddit', source_data2)
source_graph.extend(source_graph2)


source_data3, source_text3, source_classes3, source_c_descs3 = load_data('arxiv_2023', seed=0)
source_graph3 = parse_data('arxiv_2023', source_data3)
source_graph.extend(source_graph3)


source_data4, source_text4, source_classes4, source_c_descs4 = load_data('ogbn-products', seed=0)
source_graph4 = parse_data('ogbn-products', source_data4)
source_graph.extend(source_graph4)

source_data5, source_text5, source_classes5, source_c_descs5 = load_data('pubmed', seed=0)
source_graph5 = parse_data('pubmed', source_data5)
source_graph.extend(source_graph5)

print(f"We have {len(source_graph)} pretraining graphs")

batch_size= 800*7#700*8 # 800*8

train_loader = DataListLoader(source_graph, batch_size=batch_size, shuffle=True) # use DataListLoader for DP rather than DataLoader

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
    _, _, target_test_loader = split_dataloader(data, target_graph, batch_size)
    
    target_test_loaders.append(target_test_loader)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--attn_type', default='multihead',
    help="Global attention type such as 'multihead' or 'performer'.")
args = parser.parse_args()



def calculate_loss(graph_logits, text_logits, criterion):
    batch_size = graph_logits.shape[0]
    gt = torch.arange(batch_size).to(device)
    total_train_image_loss = criterion(graph_logits, gt)
    total_train_text_loss = criterion(text_logits, gt)
    
    total_train_loss = (total_train_image_loss + total_train_text_loss)/2
    return total_train_loss

class TextCLIP(torch.nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model
        
    def forward(self,input_ids, token_type_ids, attention_mask):
        # return self.model.encode_text(input_ids, token_type_ids, attention_mask)
        return self.model.encode_text(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    
class GCLIP(torch.nn.Module):
    def __init__(self, model) :
        super(GCLIP, self).__init__()
        self.model = model
        
    def forward(self,batch):
        return self.model.encode_graph(batch)


def create_logits(x1,x2,logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 =  logit_scale*x1 @ x2.t()
    logits_per_x2 =  logit_scale*x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2

attn_kwargs = {'dropout': 0.0} # 0.3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
text_model="tiny"
model = GraphCLIP(384, 1024, 12, attn_kwargs, text_lora=False, text_model=text_model)

# freeze text model
model.freeze_text()

# DP codes
model_text = TextCLIP(model)
model_graph = GCLIP(model)
model_text = torch.nn.DataParallel(model_text) # use torch DP
model_graph = DataParallel(model_graph) # use pyg DP
model.to(device)

text_ids = {
    'tiny': 'sentence-transformers/all-MiniLM-L6-v2', 
    'sbert': 'sentence-transformers/multi-qa-distilbert-cos-v1', # , # 'sentence-transformers/multi-qa-distilbert-cos-v1',
    'e5': 'intfloat/e5-base-v2'
}

tokenizer = AutoTokenizer.from_pretrained(text_ids[text_model])
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,model.parameters()), lr=1e-5, weight_decay=1e-5)

scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20,
                              min_lr=0.00001)


num_epochs=50
# lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[num_epochs//3], gamma=0.1)
# warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

def train(data_loader):
    total_loss = 0
    criterion  = torch.nn.CrossEntropyLoss()
    for batch in data_loader:
        optimizer.zero_grad()
        model.train()
        
        model.graph_model.redraw_projection.redraw_projections()
        summaries = [g.summary for g in batch]
        batch_t = tokenizer(summaries, truncation=True, padding=True, return_tensors="pt", max_length=512)
        # DP codes
        batch = [graph_aug(g, 0.4, 0.0) for g in batch] 

        def node_attack(perturbs):
            for b_id, g in enumerate(batch):
                g.x += perturbs[b_id]
            graph_embs, center_embs = model_graph(batch)
            text_embs = model_text(input_ids=batch_t['input_ids'], token_type_ids=None, attention_mask=batch_t['attention_mask'])
            
            logit_scale = model.logit_scale.exp()
            logits_per_graph, logits_per_text = create_logits(graph_embs, text_embs, logit_scale)
            loss = calculate_loss(logits_per_graph, logits_per_text, criterion)
            return loss
        perturb_shapes = [g.x.shape for g in batch]
        loss = adversarial_aug_train(model_graph, model_text, node_attack, perturb_shapes, 1e-2, 3)

        loss.backward()
        total_loss += loss.item() * len(batch)
        optimizer.step()

        # with warmup_scheduler.dampening():
        #     if i + 1 == len(data_loader):
        #         lr_scheduler.step()
    return total_loss / len(data_loader.dataset)


print(f"Let's use {torch.cuda.device_count()} GPUs!")


for epoch in range(1, num_epochs):
    loss = train(train_loader)
    res_str = f"Epoch: {epoch:02d}, Loss: {loss:.4f},"
    torch.save(model.state_dict(), f="./ckpts/graphclip.pt")


