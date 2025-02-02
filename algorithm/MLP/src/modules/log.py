"""
Author: Antonello Paolino
Date: 2025-01-31
Description:    Module for logging data and output
"""

import wandb
import sys
import numpy as np


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
    dataset_list,
    model,
    device,
    scaling,
    label,
):
    print(f"Computing and logging {label} aerodynamic forces error")
    aero_force_errors = np.zeros((len(dataset_list), 3))
    for i, sim in enumerate(dataset_list):
        # Get input: v_x, v_y, v_z, x, y, z
        input = sim[:, [0, 1, 2, 22, 23, 24]].float().to(device)
        # Compute forward pass
        model.to(device)
        model.eval()
        output = model(input).detach().cpu().numpy()
        # Rescale output
        press_coeff = output[:, 0] * (scaling[8] - scaling[7]) + scaling[7]
        x_fric_coeff = output[:, 1] * (scaling[10] - scaling[9]) + scaling[9]
        y_fric_coeff = output[:, 2] * (scaling[10] - scaling[9]) + scaling[9]
        z_fric_coeff = output[:, 3] * (scaling[10] - scaling[9]) + scaling[9]
        # Compute centroidal aerodynamic force
        dyn_press = 0.5 * 1.225 * 17.0**2
        face_normals = sim[:, 25:28]
        areas = sim[:, 32]
        pressure = press_coeff * dyn_press
        shear_x = x_fric_coeff * dyn_press
        shear_y = y_fric_coeff * dyn_press
        shear_z = z_fric_coeff * dyn_press
        d_force_x = pressure * face_normals[:, 0] * areas + shear_x * areas
        d_force_y = pressure * face_normals[:, 1] * areas + shear_y * areas
        d_force_z = pressure * face_normals[:, 2] * areas + shear_z * areas
        pred_aero_force = np.array(
            [np.sum(d_force_x), np.sum(d_force_y), np.sum(d_force_z)]
        )
        # Rescale dataset flow variables
        cp_data = sim[:, 28] * (scaling[8] - scaling[7]) + scaling[7]
        fx_data = sim[:, 29] * (scaling[10] - scaling[9]) + scaling[9]
        fy_data = sim[:, 30] * (scaling[10] - scaling[9]) + scaling[9]
        fz_data = sim[:, 31] * (scaling[10] - scaling[9]) + scaling[9]
        # Compute dataset aerodynamic force
        pressure = cp_data * dyn_press
        shear_x = fx_data * dyn_press
        shear_y = fy_data * dyn_press
        shear_z = fz_data * dyn_press
        d_force_x = pressure * face_normals[:, 0] * areas + shear_x * areas
        d_force_y = pressure * face_normals[:, 1] * areas + shear_y * areas
        d_force_z = pressure * face_normals[:, 2] * areas + shear_z * areas
        dataset_aero_force = np.array(
            [np.sum(d_force_x), np.sum(d_force_y), np.sum(d_force_z)]
        )
        # Compute error
        aero_force_errors[i, :] = np.abs(dataset_aero_force - pred_aero_force)
    # Compute Mean Squared Error
    aero_force_mse = np.mean(aero_force_errors**2, axis=0)
    # Display MSE
    print(f"MSE Drag Force ({label}): {aero_force_mse[2]}")
    print(f"MSE Lift Force ({label}): {aero_force_mse[1]}")
    print(f"MSE Side Force ({label}): {aero_force_mse[0]}")
    # Log global errors
    wandb.log(
        {
            f"MSE Drag Force ({label})": aero_force_mse[2],
            f"MSE Lift Force ({label})": aero_force_mse[1],
            f"MSE Side Force ({label})": aero_force_mse[0],
        }
    )
