"""
Author: Antonello Paolino
Date: 2025-01-31
Description:    Module for the learning architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from modules.constants import Const


# Multi Layer Perceptron
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(Const.in_dim, Const.hid_dim)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(Const.hid_dim, Const.hid_dim),
                    nn.ReLU(),
                    nn.Dropout(p=Const.dropout),
                )
                for _ in range(Const.hid_layers)
            ]
        )
        self.output_layer = nn.Linear(Const.hid_dim, Const.out_dim)

    def forward(self, input):
        input_layer_out = F.relu(self.input_layer(input))
        for hidden_layer in self.hidden_layers:
            input_layer_out = hidden_layer(input_layer_out)
        output = self.output_layer(input_layer_out)
        return output


class MlpDataset(Dataset):
    def __init__(self, X, y, batch_size):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(X)
        self.Y = torch.tensor(y)
        self.batch_size = batch_size
        self.num_batches = (len(self.X) + batch_size - 1) // batch_size

    def __len__(self):
        # this should return the size of the dataset
        return self.num_batches

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.X))
        return self.X[start:end], self.Y[start:end]
