"""
Author: Antonello Paolino
Date: 2025-01-31
Description: Module for logging data and output
"""

import wandb
import torch
import sys
import numpy as np

from modules.constants import Const


def init_wandb_project(options):
    if options["entity"] is None and options["project"] is None:
        print(
            "\n\033[31m No entity or project provided for wandb.\nKilling execution \033[0m"
        )
        sys.exit()
    runs = wandb.Api().runs(
        f"{options['entity']}/{options['project']}", order="-created_at"
    )
    if len(runs) == 0:
        run_name = "trial-1"
    else:
        run_name = "trial-" + str(int(runs[0].name.split("-")[-1]) + 1)
    wandb.init(project=options["project"], name=run_name, config=options)
    return run_name


def log_aerodynamic_forces_error(
    dataset,
    scaled_dataset,
    model,
    device,
    scaling,
    label,
):
    print(f"Computing and logging {label} aerodynamic forces error")
    aero_force_errors = np.zeros((len(dataset), 3))
    for i, zip_data in enumerate(zip(dataset, scaled_dataset)):
        data, scaled_data = zip_data
        # Get input: v_x, v_y, v_z, x, y, z, n_x, n_y, n_z
        if Const.in_dim == 6:
            x = scaled_data.x[:, Const.vel_idx + Const.pos_idx]
        elif Const.in_dim == 9:
            x = scaled_data.x[:, Const.vel_idx + Const.pos_idx + Const.face_normal_idx]
        # Compute forward pass
        model.to(device)
        model.eval()
        output = model(x.to(device), data.edge_index.to(device)).detach().cpu().numpy()
        # Rescale output
        if Const.scale_mode == "minmax":
            output = output * (scaling[4] - scaling[3]) + scaling[3]
        elif Const.scale_mode == "standard":
            output = output * scaling[4] + scaling[3]
        press_coeff = output[:, 0]
        fric_coeff = output[:, 1:]
        # Compute predicted centroidal aerodynamic force
        dyn_press = 0.5 * 1.225 * 17.0**2
        face_normals = data.x[:, Const.face_normal_idx].numpy()
        areas = data.x[:, Const.area_idx].numpy()
        pressure = press_coeff.reshape(-1, 1) * dyn_press
        friction = fric_coeff * dyn_press
        d_force = pressure * face_normals * areas + friction * areas
        pred_aero_force = np.sum(d_force, axis=0)
        # Compute dataset aerodynamic force
        pressure = data.y[:, Const.flow_idx[0]].numpy().reshape(-1, 1) * dyn_press
        friction = data.y[:, Const.flow_idx[1:]].numpy() * dyn_press
        d_force = pressure * face_normals * areas + friction * areas
        dataset_aero_force = np.sum(d_force, axis=0)
        # Compute error
        aero_force_errors[i, :] = np.abs(dataset_aero_force - pred_aero_force)
    # Compute Mean Squared Error
    aero_force_rmse = np.sqrt(np.mean(aero_force_errors**2, axis=0))
    # Display MSE
    print(f"RMSE X Force ({label}): {aero_force_rmse[0]}")
    print(f"RMSE Y Force ({label}): {aero_force_rmse[1]}")
    print(f"RMSE Z Force ({label}): {aero_force_rmse[2]}")
    # Log global errors
    wandb.log(
        {
            f"RMSE X Force ({label})": aero_force_rmse[0],
            f"RMSE Y Force ({label})": aero_force_rmse[1],
            f"RMSE Z Force ({label})": aero_force_rmse[2],
        }
    )
