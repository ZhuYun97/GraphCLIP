from torch_geometric.loader import DataListLoader, DataLoader

# for data parallel
def split_dataloader(data, graphs, batch_size):
    train_idx = data.train_mask.nonzero().squeeze()
    val_idx = data.val_mask.nonzero().squeeze()
    test_idx = data.test_mask.nonzero().squeeze()
    train_dataset = [graphs[idx] for idx in train_idx]
    val_dataset = [graphs[idx] for idx in val_idx]
    test_dataset = [graphs[idx] for idx in test_idx]

    train_loader = DataListLoader(train_dataset, batch_size=batch_size, shuffle=True) # use DataListLoader for DP rather than DataLoader
    val_loader = DataListLoader(val_dataset, batch_size=batch_size)
    test_loader = DataListLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def split_dataloader_singlecard(data, graphs, batch_size):
    train_idx = data.train_mask.nonzero().squeeze()
    val_idx = data.val_mask.nonzero().squeeze()
    test_idx = data.test_mask.nonzero().squeeze()
    train_dataset = [graphs[idx] for idx in train_idx]
    val_dataset = [graphs[idx] for idx in val_idx]
    test_dataset = [graphs[idx] for idx in test_idx]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # use DataListLoader for DP rather than DataLoader
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader