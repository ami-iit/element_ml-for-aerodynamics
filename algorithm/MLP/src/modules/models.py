"""
Author: Antonello Paolino
Date: 2025-01-31
Description:    Module for the learning architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from modules import glob


# Multi Layer Perceptron
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(glob.in_dim, glob.hid_dim)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(glob.hid_dim, glob.hid_dim),
                    nn.ReLU(),
                    nn.Dropout(p=glob.dropout),
                )
                for _ in range(glob.hid_layers)
            ]
        )
        self.output_layer = nn.Linear(glob.hid_dim, glob.out_dim)

    def forward(self, input):
        input_layer_out = F.relu(self.input_layer(input))
        for hidden_layer in self.hidden_layers:
            input_layer_out = hidden_layer(input_layer_out)
        output = self.output_layer(input_layer_out)
        return output


class MlpDataset(Dataset):
    def __init__(self, X, y):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # this should return the size of the dataset
        return len(self.X)

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        features = self.X[idx]
        target = self.Y[idx]
        return features, target
