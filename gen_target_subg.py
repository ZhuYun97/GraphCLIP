from data.load import load_data
from data.sampling import ego_graphs_sampler, pyg_random_walk
import torch
from tqdm import tqdm
import json
from utils.args import Arguments


if __name__ == '__main__':
    config = Arguments().parse_args()

    data, text, _, _ = load_data(config.dataset, seed=0)


    node_ids = torch.arange(data.y.shape[0])
    if config.sampler == "rw":
        all_n_ids, all_edges = pyg_random_walk(node_ids, data, length=config.walk_steps, restart_prob=config.restart)
    elif config.sampler == 'khop':
        graphs = ego_graphs_sampler(node_ids, data, hop=config.k) # 3 for citeseer, cora
    

    item_list = []
    for i, seed in tqdm(enumerate(node_ids.tolist())):
        item = {}
        item['id'] = seed
        if config.sampler == 'rw':
            item['graph'] = all_edges[i].tolist()
        elif config.sampler == 'khop':
            item['graph'] = graphs[i].edge_index.tolist()
        
        item['summary'] = ""  # we do not  need to generate summaries for target datasets
        item_list.append(item)

    with open(f'./target_data/{config.dataset}.json', 'w', encoding='utf-8') as f:
        json.dump(item_list, f, indent=4, ensure_ascii=False)