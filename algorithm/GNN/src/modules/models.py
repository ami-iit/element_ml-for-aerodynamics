"""
Author: Antonello Paolino
Date: 2025-01-31
Description: Module for the learning architectures
"""

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from modules.constants import Const


class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(Const.in_dim, Const.enc_dim),  # Input layer
            nn.ReLU(),
            *[  # Hidden layers
                nn.Sequential(
                    nn.Linear(Const.enc_dim, Const.enc_dim),
                    nn.ReLU(),
                    nn.Dropout(p=Const.dropout),
                )
                for _ in range(Const.enc_layers)
            ],
            nn.Linear(Const.enc_dim, Const.latent_dim),  # Output layer
        )
        # GNN layers
        self.conv_layers = nn.ModuleList(
            [
                gnn.GCNConv(Const.latent_dim, Const.latent_dim)
                for _ in range(Const.gnc_layers)
            ]
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(Const.latent_dim, Const.dec_dim),  # Input layer
            nn.ReLU(),
            *[  # Hidden layers
                nn.Sequential(
                    nn.Linear(Const.dec_dim, Const.dec_dim),
                    nn.ReLU(),
                    nn.Dropout(p=Const.dropout),
                )
                for _ in range(Const.dec_layers)
            ],
            nn.Linear(Const.dec_dim, Const.out_dim),  # Output layer
        )

    def forward(self, x, edge_index):
        out = self.encoder(x)
        for conv in self.conv_layers:
            out = conv(out, edge_index)
            out = F.relu(out)
            out = F.dropout(out, p=Const.dropout, training=self.training)
        output = self.decoder(out)
        return output


def initialize_weights_xavier_normal(model):
    for _, layer in model.named_modules():
        if isinstance(layer, (nn.Linear)):
            nn.init.xavier_uniform_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias) if layer.bias is not None else None
        elif isinstance(layer, (gnn.GCNConv)):
            nn.init.xavier_uniform_(layer.lin.weight, gain=1.0)
            nn.init.zeros_(layer.lin.bias) if layer.lin.bias is not None else None
