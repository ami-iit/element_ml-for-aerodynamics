"""
Author: Antonello Paolino
Date: 2025-01-31
Description:    Module for the learning architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
