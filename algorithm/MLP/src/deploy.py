"""
Author: Antonello Paolino
Date: 2025-01-31
Description:    Main module to execute the training of the Multi-Layer
                Perceptron for Aerodynamics prediction of iRonCub surface
                flow variables (MLP-Aero).
"""

import numpy as np
import torch
import random as random
import time as time
from pathlib import Path

from modules.constants import Const

from modules.run import Run

RUN_NAME = "trial-11"
SCALE_MODE = "standard"  # "standard", "minmax"
PITCH = 40.0
YAW = 0.0
WIND_SPEED = 17.0


def main():
    # Get root directory path
    root = Path(__file__).parents[1]

    # Set the database path
    dataset_dir = root / "test_cases" / "ironcub" / "database"
    dataset_file = dataset_dir / "ironcub-hovering.npz"

    # Initialize the run object
    run = Run()
    if RUN_NAME == "trial-9":
        run.load_local_model(root.parents[2] / "artifacts" / "model-v9")
    else:
        run.load_train_from_wandb("ami-iit/MLP-iRonAero", RUN_NAME)
    run.load_dataset(dataset_file)

    # Compute aerodynamic forces MSE
    run.compute_aerodynamic_forces(WIND_SPEED, SCALE_MODE)

    # Visualization
    # Compute dataset sample
    sample = np.where((run.pitch_angles == PITCH) & (run.yaw_angles == YAW))[0][0]
    points = run.dataset[sample][:, Const.pos_idx]
    values = run.dataset[sample][:, Const.flow_idx[0]]
    print(f"Database aerodynamic force: {run.aero_forces_in[sample]}")
    # Compute prediction with the model using scaled input
    input_vel = run.dataset[sample][:, Const.vel_idx] / run.scaling[0]
    if SCALE_MODE == "minmax":
        input_pos = (points - run.scaling[1]) / (run.scaling[2] - run.scaling[1])
    elif SCALE_MODE == "standard":
        input_pos = (points - run.scaling[1]) / run.scaling[2]
    input = np.concatenate((input_vel, input_pos), axis=1)
    input = torch.from_numpy(input).float().to(run.device)
    output = run.model(input)
    output = output.cpu().detach().numpy()
    if SCALE_MODE == "minmax":
        output = output * (run.scaling[4] - run.scaling[3]) + run.scaling[3]
    elif SCALE_MODE == "standard":
        output = output * run.scaling[4] + run.scaling[3]
    print(f"Predicted aerodynamic force: {run.aero_forces_out[sample]}")
    # Visualize pointclouds from the database and the prediction
    run.visualize_pointcloud(points, values, window_name="database sample")
    run.visualize_pointcloud(points, output[:, 0], window_name="MLP prediction")


if __name__ == "__main__":
    main()
