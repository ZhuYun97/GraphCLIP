import torch
from torch_geometric import seed_everything
from transformers import AutoTokenizer

from data.load import load_data
from models import GraphCLIP
from utils.args import Arguments
from utils.process import parse_target_data, split_dataloader


eval_template={
    'cora': "this paper has a topic on {c}", 
    'citeseer': "good paper of {c} ", 
    'pubmed': "it belongs to {c} research area",
    'arxiv_2023': "it belongs to {c} research area",
    'wikics': "it belongs to {c} research area",
    'photo':  "this product belongs to {c}",
    'computer':  "is {c} category", 
    'history': "this book belongs to {c}",
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

if __name__ == "__main__":
    config = Arguments().parse_args()
    seed_everything(88) 
    attn_kwargs = {'dropout': 0.0}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphCLIP(384, 1024, 12, attn_kwargs, text_model=config.lm_type)
    model.load_state_dict(torch.load("./checkpoints/graphclip.pt"), strict=False)
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model.to(device)
    print("mdoel is loaded")
    
    ################ target data
    target_data = config.target_data.split("*") # testing citeseer dataset, you can add more datasets here
    target_datasets = target_data
    target_classes_list = []
    target_c_desc_list = []
    target_test_loaders = []
    for d in target_data:
        data, text, classes, c_descs = load_data(d, seed=0)
        target_classes_list.append(classes)
        target_c_desc_list.append(c_descs)
        target_graph = parse_target_data(d, data)
        _, _, target_test_loader = split_dataloader(data, target_graph, config.batch_size, seed=0,name=d)
        
        target_test_loaders.append(target_test_loader)
        print(f"{d} is loaded")
    
    res_str = ""
    all_test_list = []

    run_test = []
    for i, classes in enumerate(target_classes_list):
        test_acc = test(target_test_loaders[i], classes, target_c_desc_list[i], target_datasets[i])
        run_test.append(test_acc)
        res_str += f" {target_datasets[i]} acc: {test_acc}"
    print(1, res_str)

