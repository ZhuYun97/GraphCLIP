import json
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataListLoader
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.nn import DataParallel
from transformers import AutoTokenizer

from data.load import load_data
from models import GraphCLIP
from utils.augmentation import adversarial_aug_train, graph_aug
from utils.args import Arguments
from utils.process import parse_source_data
from models.dp import TextCLIP, GCLIP, calculate_loss, create_logits


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
        batch = [graph_aug(g, 0.3, 0.2) for g in batch] 

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

    return total_loss / len(data_loader.dataset)
    
if __name__ == "__main__":
    config = Arguments().parse_args()
    seed_everything(88) 
    attn_kwargs = {'dropout': 0.0}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_model = config.lm_type
    model = GraphCLIP(384, 1024, 12, attn_kwargs, text_model=text_model)

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
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
    
    # collect source data
    source_name_list = config.source_data.split("+")
    all_source_graph = []
    for source_name in source_name_list:
        source_data, source_text, source_classes, source_c_descs = load_data(source_name)
        source_graph = parse_source_data(source_name, source_data)
        all_source_graph.extend(source_graph)
    
    print(f"We have {len(all_source_graph)} pretraining graphs")

    train_loader = DataListLoader(source_graph, batch_size=config.batch_size, shuffle=True) # use DataListLoader for DP rather than DataLoader

    print(f"Let's use {torch.cuda.device_count()} GPUs!")

    for epoch in range(1, config.epochs):
        loss = train(train_loader)
        res_str = f"Epoch: {epoch:02d}, Loss: {loss:.4f},"
        print(res_str)
        torch.save(model.state_dict(), f="./checkpoints/graphclip.pt")


