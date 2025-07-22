"""
Author: Antonello Paolino
Date: 2025-01-31
Description: Module for the learning architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import wandb

from modules.constants import Const


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) model:
    - GNN layers with GCNConv for graph convolution
    """

    def __init__(self):
        super(GCN, self).__init__()
        # GCN layers
        self.conv_layers = nn.ModuleList(
            [gnn.GCNConv(Const.in_dim, Const.latent_dim, normalize=False)]
            + [
                gnn.GCNConv(Const.latent_dim, Const.latent_dim, normalize=False)
                for _ in range(Const.gnc_layers - 2)
            ]
            + [gnn.GCNConv(Const.latent_dim, Const.out_dim, normalize=False)]
        )

    def forward(self, x, edge_index):
        # Input layer
        out = self.conv_layers[0](x, edge_index)
        out = F.relu(out)
        out = F.dropout(out, p=Const.dropout, training=self.training)
        # Hidden layers
        for conv in self.conv_layers[1:-1]:
            out = conv(out, edge_index)
            out = F.relu(out)
            out = F.dropout(out, p=Const.dropout, training=self.training)
        # Output layer
        output = self.conv_layers[-1](out, edge_index)
        return output


class HGN(nn.Module):
    """
    Hybrid Graph Network (HGN) model:
    input -> encoder -> GNC -> decoder -> output
    - MLP with homogeneus layer dimensions for encoder and decoder
    - GNN layers with GCNConv for graph convolution
    """

    def __init__(self):
        super(HGN, self).__init__()
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
                gnn.GCNConv(Const.latent_dim, Const.latent_dim, normalize=False)
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


class GAE(nn.Module):
    """
    Graph AutoEncoder (GAE) model:
    input -> encoder -> GNC -> decoder -> output
    - MLP with decreasing/increasing layer dimensions for encoder and decoder
    - GNN layers with GCNConv for graph convolution
    """

    def __init__(self):
        super(GAE, self).__init__()
        # Encoder layers
        rate = (Const.enc_dim - Const.latent_dim) / (Const.enc_layers - 1)
        enc_dims = [int(Const.enc_dim - rate * i) for i in range(Const.enc_layers)]
        self.encoder = nn.Sequential(
            nn.Linear(Const.in_dim, Const.enc_dim),  # Input layer
            nn.ReLU(),
            *[  # Hidden layers
                nn.Sequential(
                    nn.Linear(enc_dims[i], enc_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(p=Const.dropout),
                )
                for i in range(Const.enc_layers - 1)
            ],
        )
        # GNN layers
        self.conv_layers = nn.ModuleList(
            [
                gnn.GCNConv(Const.latent_dim, Const.latent_dim, normalize=False)
                for _ in range(Const.gnc_layers)
            ]
        )
        # Decoder layers
        rate = (Const.dec_dim - Const.latent_dim) / (Const.enc_layers - 1)
        dec_dims = [int(Const.latent_dim + rate * i) for i in range(Const.dec_layers)]
        self.decoder = nn.Sequential(
            *[  # Hidden layers
                nn.Sequential(
                    nn.Linear(dec_dims[i], dec_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(p=Const.dropout),
                )
                for i in range(Const.dec_layers - 1)
            ],
            nn.Linear(Const.dec_dim, Const.out_dim),  # Output layer
            nn.ReLU(),
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


def load_wandb_model(model, optimizer):
    # Get model checkpoint from wandb
    api = wandb.Api()
    model_artifact = api.artifact(Const.project + "/model:" + Const.trial_name)
    checkpoint = torch.load(model_artifact.download() + r"/ckp_model.pt")
    # Load model and set weights
    model.load_state_dict(checkpoint["model_state"])
    # Load optimizer and set state
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return model, optimizer
