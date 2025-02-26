from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from utils.args import Arguments
from data.load import load_data
from data.sampling import pyg_random_walk
import torch
from tqdm import tqdm
import json




research_areas = {"obgn-arxiv": "arXiv CS sub-category",
                  "arxiv_2023": "arXiv CS sub-category",
                 "pubmed": "specific research areas (e.g., 'Diabetes Mellitus Experimental', 'Diabetes Mellitus Type1', 'Diabetes Mellitus Type2')",
                 'ogbn-products': "1) Home & Kitchen, 2) Health & Personal Care, 3) Beauty, 4) Sports & Outdoors, 5) Books, 6) Patio, Lawn & Garden, 7) Toys & Games, 8) CDs & Vinyl, 9) Cell Phones & Accessories, 10) Grocery & Gourmet Food, 11) Arts, Crafts & Sewing, 12) Clothing, Shoes & Jewelry, 13) Electronics, 14) Movies & TV, 15) Software, 16) Video Games, 17) Automotive, 18) Pet Supplies, 19) Office Products, 20) Industrial & Scientific, 21) Musical Instruments, 22) Tools & Home Improvement, 23) Magazine Subscriptions, 24) Baby Products, 25) NAN, 26) Appliances, 27) Kitchen & Dining, 28) Collectibles & Fine Art, 29) All Beauty, 30) Luxury Beauty, 31) Amazon Fashion, 32) Computers, 33) All Electronics, 34) Purchase Circles, 35) MP3 Players & Accessories, 36) Gift Cards, 37) Office & School Supplies, 38) Home Improvement, 39) Camera & Photo, 40) GPS & Navigation, 41) Digital Music, 42) Car Electronics, 43) Baby, 44) Kindle Store, 45) Kindle Apps, 46) Furniture & Decor",
                 'reddit': "'Normal Users', 'Popular Users'"
                 }

# this prompt is used for ogbn-products
# prompt = """I have a GraphML file representing an Amazon product co-purchasing network. In this network, nodes represent products sold on Amazon, edges indicate that two products are frequently purchased together.

# I would like you to analyze the product represented by the node 'n{seed}' using the GraphML data in the following two ways:

# ### 1. Product Summary and Context Analysis
# - Extract and summarize the details of the product denoted by 'n{seed}', including its title and description (if available).
# - Provide an overall summary of the prevalent themes or trends among the products that co-purchased with 'n{seed}'. Identify common threads or topics shared by these neighboring products.

# ### 2. Category Classification
# - Using the information gathered from 'n{seed}' and its neighboring nodes, classify 'n{seed}' into one of the specified categories: {categories}. Please think step by step.

# ### Constraints
# - Your analysis should be directly based on the data provided in the GraphML file and should be limited to 400 tokens. Focus exclusively on node 'n{seed}' and its immediate co-purchased neighborhood.

# **GraphML Co-purchasing Network Data:**
# """

# this prompt is used for citation network
prompt = """I am providing you with a GraphML file depicting a citation network in artificial intelligence research. Each node in the network represents a scholarly article, and each edge signifies a citation relationship between articles. Please analyze the article represented by node 'n{seed}' using the provided GraphML data in the following two ways:

1. Paper Summary and Context Analysis:
Extract and summarize the key findings or contributions of the paper denoted by 'n{seed}'. Consider the details embedded within node 'n{seed}', including its title, abstract, and keywords (if available).
Provide an overall summary of prevalent themes or concepts shared by the papers that cite or are cited by 'n{seed}' (its direct neighbors in the network). Identify common threads or research topics among these neighbors.

2. Research Area Classification:
Based on the information summarized from 'n{seed}' and its neighboring nodes, determine the one of {research_area} to which 'n{seed}' primarily contributes.
Justify the classification by explaining which aspects of 'n{seed}' align with recognized themes, issues, or methodologies in the identified research area(s).

Please ensure your analyses are grounded in the data provided by the GraphML file within 400 tokens, focusing on node 'n{seed}' and its immediate citation neighborhood. The detailed GraphML citation network data is as follows:
"""

def trans_graph_code(sub_n_id, sub_e_id, node_map):
    graphML="""<?xml version="1.0" encoding="UTF-8"?>
    <graphml>
        <key id="k0" for="node" attr.name="title" attr.type="string">
            <default>unkown</default>
        </key>
        <key id="k1" for="node" attr.name="description" attr.type="string"/>
        <key id="k2" for="edge" attr.name="relation" attr.type="string"/>
        <graph id="G" edgedefault="undirected">
    {nodes}
    {edges}
        </graph>
    </graphml>
    """

    node_idx = sub_n_id
    papers = []
    abs = []
    for i in node_idx:
        papers.append(title_list[i])
        abs.append(abs_list[i])

    sources_idx, target_idx = sub_e_id
    node_str = ""
    edge_str = ""
    for i,p in enumerate(papers):
        tmp = f'\t<node id="n{sub_n_id[i]}">\n\t\t<data key="k0">{p}</data>\n\t\t<data key="k1">{abs[i]}</data>\n\t</node>'
        # tmp = f'\t<node id="n{node_map[sub_n_id[i]]}">\n\t\t<data key="k0">{p}</data>\n\t\t<data key="k1"><graph></data>\n\t</node>'
        if i != len(papers)-1:
            tmp += "\n"
        node_str += tmp

    for i, e in enumerate(sources_idx):
        src_id = sources_idx[i].item()
        dst_id = target_idx[i].item()
        tmp = f'\t<edge id="e{i}" source="n{src_id}" target="n{dst_id}">\n\t\t<data key="k2">co-purchasing</data>\n\t</edge>'
        # tmp = f'\t<edge id="e{i}" source="n{node_map[src_id]}" target="n{node_map[dst_id]}">\n\t\t<data key="k2">cites</data>\n\t</edge>'
        if i != len(sources_idx):
            tmp += '\n'
        edge_str += tmp

    graphML_str = graphML.format(nodes=node_str, edges=edge_str)
    return graphML_str

def generate_summary_vllm(prompts, model, max_tokens=512, temperature=0.8, top_p=0.95):
    stop_token_ids = [151329, 151336, 151338]
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stop_token_ids)
    outputs = model.generate(prompts, sampling_params)
    return outputs

if __name__ == '__main__':
    config = Arguments().parse_args()

    data, text, num_classes, class_string = load_data(config.dataset, use_text=True, seed=0)

    node_ids = torch.arange(data.x.shape[0])
    # node_ids = data.train_mask.nonzero().squeeze() # only use training data (for ogbn-products)
    all_n_ids, all_edges = pyg_random_walk(node_ids, data, length=config.walk_steps, restart_prob=config.restart)

    model_id = "~/Qwen2-72B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = LLM(model=model_id, max_model_len=15000,trust_remote_code=True, dtype=torch.bfloat16, tensor_parallel_size=8,gpu_memory_utilization=0.91) # 8*A100(40G)


    # split passage into title and abstract
    title_list = []
    abs_list = []
    for passage in text:
        splited = passage.split('\n')
        assert len(splited) >= 2
        title, abs = splited[0], splited[1]
        title_list.append(title)
        abs_list.append(abs.strip())

    item_list = []
    messages = []
    for i, seed in tqdm(enumerate(node_ids.tolist())):
        item = {}
        item['id'] = seed
        item['graph'] = all_edges[i].tolist()
        sub_n_id = torch.unique(all_n_ids[i]).tolist()
        sub_e_id = all_edges[i]
        node_map = {n:j for j, n in enumerate(sub_n_id)} # relative mapping

        prompt_str = prompt.format(seed=seed, categories=research_areas[config.dataset])
        graph_code = trans_graph_code(sub_n_id, sub_e_id, node_map) # encode graph into sequence

        instrcution = prompt_str + graph_code
        message = [{
            "role": "system",
            "content": "You are a helpful language and graph assistant. You are able to understand the graph content that the user provides, and assist the user with a variety of tasks using natural language."
        },
        {
            "role": "user",
            "content": instrcution
        }]
        messages.append(message)
        item_list.append(item)

    # start generating graph summary
    batch=32 
    for i in tqdm(range(0, len(messages), batch)):
        if i+batch>= len(messages):
            end = len(messages)
        else:
            end = i+batch
        cur_messages = messages[i:end]
        cur_messages = tokenizer.apply_chat_template(cur_messages, tokenize=False, add_generation_prompt=True)
        responses = generate_summary_vllm(cur_messages, model, max_tokens=512)

        for j, res in enumerate(responses):
            item_list[i+j]['summary'] = res.outputs[0].text

    with open(f'summary/summary-{config.dataset}.json', 'w', encoding='utf-8') as f:
        json.dump(item_list, f, indent=4, ensure_ascii=False)