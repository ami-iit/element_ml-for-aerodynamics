"""
Author: Antonello Paolino
Date: 2025-01-31
Description: Module for the learning architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import grad
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

    def forward(self, pred, target, normals, areas):
        force_loss, i, end_idx = 0, 0, 0
        while end_idx < target.shape[1] - 1:
            # for i in range(Const.batch_size):
            start_idx, end_idx = i * Const.sim_len, (i + 1) * Const.sim_len - 1
            d_fa_press = torch.sum(
                (pred[0, start_idx:end_idx, 0] - target[0, start_idx:end_idx, 0])[
                    :, None
                ]
                * normals[0, start_idx:end_idx, :]
                * areas[0, start_idx:end_idx, :],
                dim=0,
            )
            d_fa_shear = torch.sum(
                (pred[0, start_idx:end_idx, 1:] - target[0, start_idx:end_idx, 1:])
                * areas[0, start_idx:end_idx, :],
                dim=0,
            )
            force_loss += torch.norm(d_fa_press + d_fa_shear, p=2)
            i += 1
        force_loss /= Const.batch_size

        # Total loss
        return force_loss


class PhysicsInformedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, output, target, input):
        """
        output: model output (p, tau_11, tau_12, tau_13, tau_22, tau_23, tau_33)
        target: ground truth values (p, tau_w,x, tau_w,y, tau_w,z)
        input: model input (v_x, v_y, v_z, x, y, z, n_x, n_y, n_z)
        """
        p, tau11, tau12, tau13, tau22, tau23, tau33 = [output[:, i] for i in range(7)]
        n_x, n_y, n_z = input[:, 6], input[:, 7], input[:, 8]

        # 1. Wall stress loss (data loss)
        tau_w = torch.stack(
            [
                p,
                tau11 * n_x + tau12 * n_y + tau13 * n_z,
                tau12 * n_x + tau22 * n_y + tau23 * n_z,
                tau13 * n_x + tau23 * n_y + tau33 * n_z,
            ],
            dim=1,
        )
        data_loss = self.mse(tau_w, target)

        # 2. Physics-informed loss: momentum balance
        input = input.requires_grad_(True)
        grad_out = torch.ones_like(p)

        # Compute gradients
        def dfdx(f, dim):
            return grad(f, input, grad_outputs=grad_out, create_graph=True)[0][:, dim]

        dp_dx = dfdx(p, 3)
        dp_dy = dfdx(p, 4)
        dp_dz = dfdx(p, 5)
        dt11_dx = dfdx(tau11, 3)
        dt12_dy = dfdx(tau12, 4)
        dt13_dz = dfdx(tau13, 5)
        dt12_dx = dfdx(tau12, 3)
        dt22_dy = dfdx(tau22, 4)
        dt23_dz = dfdx(tau23, 5)
        dt13_dx = dfdx(tau13, 3)
        dt23_dy = dfdx(tau23, 4)
        dt33_dz = dfdx(tau33, 5)

        # Residuals of momentum equations
        mom_x = dp_dx - (dt11_dx + dt12_dy + dt13_dz)
        mom_y = dp_dy - (dt12_dx + dt22_dy + dt23_dz)
        mom_z = dp_dz - (dt13_dx + dt23_dy + dt33_dz)

        physics_loss = (mom_x**2 + mom_y**2 + mom_z**2).mean()

        # Total loss
        return data_loss + Const.physics_informed_loss_weight * physics_loss
