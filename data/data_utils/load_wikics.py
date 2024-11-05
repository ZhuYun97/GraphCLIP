import torch
import os.path as osp


def get_raw_text_wikics(use_text=False, seed=0):
    if osp.exists(f"./processed_data/wikics.pt"):
        data = torch.load(f"./processed_data/wikics.pt", map_location='cpu')
        data.train_mask = data.train_mask[:,seed]
        data.val_mask = data.val_mask[:,seed]
        # data.test_mask = data.test_masks[seed]
        return data, data.raw_texts
    else:
        raise NotImplementedError('No existing wikics dataset!')