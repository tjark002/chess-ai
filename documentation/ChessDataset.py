import chess
import random as rd
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch

class ChessDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Load the data and split into training and validation datasets
def get_bitwise_dataset():
    container = np.load('/content/drive/MyDrive/chessai/fenboardndarray.npz', allow_pickle=True)
    x = container['x']
    y = container['y']
    y = np.asarray(y / abs(y).max() / 2 + 0.5, dtype=np.float32) # normalization

    dataset = ChessDataset(x, y)
    train_size = 80000
    val_size = 20000
    train_set, val_set = random_split(dataset, [train_size, val_size])

    return train_set, val_set

#Extract separate x_train, y_train, x_val, and y_val from the subsets
def extract_features_targets(subset):
    x_data = []
    y_data = []

    for x, y in subset:
        x_data.append(x)
        y_data.append(y)

    return torch.tensor(np.array(x_data)), torch.tensor(np.array(y_data))
