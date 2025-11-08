# dataloader.py
from torch.utils.data import DataLoader

def get_dataloader(dataset, batch_size=32, num_workers=4, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )