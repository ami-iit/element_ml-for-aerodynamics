"""
Author: Antonello Paolino
Date: 2025-01-31
Description: Module for the learning architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import wandb

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
    def __init__(self, dataset, in_idx):
        # convert into PyTorch tensors and remember them
        self.X = torch.tensor(dataset[:, in_idx])
        self.Y = torch.tensor(dataset[:, Const.flow_idx])
        self.normals = torch.tensor(dataset[:, Const.face_normal_idx])
        self.areas = torch.tensor(dataset[:, Const.area_idx])
        self.batch_size = Const.batch_size * Const.sim_len
        self.num_batches = (len(self.X) + self.batch_size - 1) // self.batch_size

    def __len__(self):
        # this should return the size of the dataset
        return self.num_batches

    def __getitem__(self, idx):
        # this should return one sample from the dataset
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.X))
        return (
            self.X[start:end],
            self.Y[start:end],
            self.normals[start:end],
            self.areas[start:end],
        )


def initialize_weights_xavier_normal(model):
    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)


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


class AeroForceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, prediction, target, normals, areas):
        # Basic MSE loss
        base_loss = self.mse(prediction, target)

        # Custom loss
        force_loss = 0
        for i in range(Const.batch_size):
            start_idx, end_idx = i * Const.sim_len, (i + 1) * Const.sim_len
            d_fa_press = torch.sum(
                (prediction[start_idx:end_idx, 0] - target[start_idx:end_idx, 0])
                * normals[start_idx:end_idx, :]
                * areas[start_idx:end_idx],
                dim=0,
            )
            d_fa_shear = torch.sum(
                (prediction[start_idx:end_idx, 1:4] - target[start_idx:end_idx, 1:4])
                * normals[start_idx:end_idx, :]
                * areas[start_idx:end_idx],
                dim=0,
            )
            force_loss += torch.norm(d_fa_press + d_fa_shear, p=2)
        force_loss /= Const.batch_size

        # Total loss
        return base_loss + Const.force_loss_weight * force_loss
