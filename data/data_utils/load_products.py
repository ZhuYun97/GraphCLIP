from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
from datasets import load_dataset


data_path = '../datasets'

def get_raw_dataset(raw_train=f"{data_path}/Amazon-3M.raw/trn.json.gz", 
                    raw_test=f"{data_path}/Amazon-3M.raw/tst.json.gz",
                    label2cat=f"{data_path}/ogbn_products/mapping/labelidx2productcategory.csv.gz",
                    idx2asin=f"{data_path}/ogbn_products/mapping/nodeidx2asin.csv.gz"):
    
    train_part = load_dataset("json", data_files=raw_train)
    test_part = load_dataset("json", data_files=raw_test)
    train_df = train_part['train'].to_pandas()
    test_df = test_part['train'].to_pandas()
    combine_df = pd.concat([train_df, test_df], ignore_index=True)
    
    label2cat_df = pd.read_csv(label2cat, compression='gzip')
    idx2asin_df = pd.read_csv(idx2asin, compression='gzip')
    
    idx_mapping = {row[0]: row[1] for row in idx2asin_df.values}
    label_mapping = {row['label idx']: row['product category'] for _, row in label2cat_df.iterrows()}
    content_mapping = {row[0]: (row[1], row[2]) for row in combine_df.values}
    
    return idx_mapping, content_mapping, label_mapping

def get_raw_text_products(use_text=False, seed=0):
    dataset = PygNodePropPredDataset(root=data_path, name='ogbn-products')
    data = dataset[0]

    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    data.y = data.y.squeeze()

    if not use_text:
        return data, None

    idx_mapping, content_mapping, label_mapping = get_raw_dataset()

    text_dict = {'title': [], 'content': [], 'label': []}
    text_list = []
    for i in range(len(data.y)):
        uid = idx_mapping.get(i, None)
        if uid:
            title, content = content_mapping.get(uid, (None, None))

            label = label_mapping.get(data.y[i].item(), None)
            
            text_dict['title'].append(title)
            text_dict['content'].append(content)

            mapped_label = products_mapping.get(label, None)
            # assert mapped_label is not None, f"Label {label} not found in mapping"
            if mapped_label is None:
                text_dict['label'].append('label 25')
            else:
                text_dict['label'].append(mapped_label)
            
            text_list.append('Title: ' + title + 'Content: ' + content)   

    return data, text_list


products_mapping = {'Home & Kitchen': 'Home & Kitchen',
        'Health & Personal Care': 'Health & Personal Care',
        'Beauty': 'Beauty',
        'Sports & Outdoors': 'Sports & Outdoors',
        'Books': 'Books',
        'Patio, Lawn & Garden': 'Patio, Lawn & Garden',
        'Toys & Games': 'Toys & Games',
        'CDs & Vinyl': 'CDs & Vinyl',
        'Cell Phones & Accessories': 'Cell Phones & Accessories',
        'Grocery & Gourmet Food': 'Grocery & Gourmet Food',
        'Arts, Crafts & Sewing': 'Arts, Crafts & Sewing',
        'Clothing, Shoes & Jewelry': 'Clothing, Shoes & Jewelry',
        'Electronics': 'Electronics',
        'Movies & TV': 'Movies & TV',
        'Software': 'Software',
        'Video Games': 'Video Games',
        'Automotive': 'Automotive',
        'Pet Supplies': 'Pet Supplies',
        'Office Products': 'Office Products',
        'Industrial & Scientific': 'Industrial & Scientific',
        'Musical Instruments': 'Musical Instruments',
        'Tools & Home Improvement': 'Tools & Home Improvement',
        'Magazine Subscriptions': 'Magazine Subscriptions',
        'Baby Products': 'Baby Products',
        'label 25': 'label 25',
        'Appliances': 'Appliances',
        'Kitchen & Dining': 'Kitchen & Dining',
        'Collectibles & Fine Art': 'Collectibles & Fine Art',
        'All Beauty': 'All Beauty',
        'Luxury Beauty': 'Luxury Beauty',
        'Amazon Fashion': 'Amazon Fashion',
        'Computers': 'Computers',
        'All Electronics': 'All Electronics',
        'Purchase Circles': 'Purchase Circles',
        'MP3 Players & Accessories': 'MP3 Players & Accessories',
        'Gift Cards': 'Gift Cards',
        'Office & School Supplies': 'Office & School Supplies',
        'Home Improvement': 'Home Improvement',
        'Camera & Photo': 'Camera & Photo',
        'GPS & Navigation': 'GPS & Navigation',
        'Digital Music': 'Digital Music',
        'Car Electronics': 'Car Electronics',
        'Baby': 'Baby',
        'Kindle Store': 'Kindle Store',
        'Buy a Kindle': 'Buy a Kindle',
        'Furniture & D&#233;cor': 'Furniture & Decor',
        '#508510': '#508510'}

products_keys_list = list(products_mapping.keys())