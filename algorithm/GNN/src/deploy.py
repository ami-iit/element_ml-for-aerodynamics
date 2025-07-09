"""
Author: Antonello Paolino
Date: 2025-01-31
Description:    Code to perform inference of the GNN model.
"""

import numpy as np
import random as random
import time as time
from pathlib import Path
from resolve_robotics_uri_py import resolve_robotics_uri

from modules.constants import Const
from modules.run import Run
from modules.robot import Robot

RUN_NAME = "trial-5"
SCALE_MODE = "standard"  # "standard", "minmax"
PITCH = 40.0
YAW = 0.0
WIND_SPEED = 17.0


def main():
    # Get root directory path
    root = Path(__file__).parents[1]

    # Set the database path
    dataset_dir = root / "test_cases" / "ironcub" / "database"
    dataset_file = dataset_dir / "ironcub-hovering-dual-graph-bodyframe.npz"

    # Initialize robot object
    robot_name = "iRonCub-Mk3"
    urdf_path = str(resolve_robotics_uri("package://iRonCub-Mk3/model.urdf"))
    robot = Robot(robot_name, urdf_path)

    # Initialize the run object
    run = Run(robot)
    run.load_dataset(dataset_file)
    run.transform_dataset()
    run.load_train_from_wandb("ami-iit/GNN-iRonAero", RUN_NAME)
    run.scale_dataset()

    # Set model input layer dimension (MLP vs MLPN)
    Const.in_dim = 9

    # Compute aerodynamic forces MSE
    run.compute_aerodynamic_forces(WIND_SPEED, SCALE_MODE)

    # Visualization
    # Compute dataset sample
    sample = np.where((run.pitch_angles == PITCH) & (run.yaw_angles == YAW))[0][0]
    points = run.dataset[sample].x[:, Const.pos_idx].numpy()
    values = run.dataset[sample].y[:, Const.flow_idx[0]].numpy()
    print(f"Database aerodynamic force: {run.aero_forces_in[sample]}")
    # Compute prediction with the model using scaled input
    x = run.scaled_dataset[sample].x[
        :, Const.vel_idx + Const.pos_idx + Const.face_normal_idx
    ]
    edge_index = run.scaled_dataset[sample].edge_index
    output = run.model(x.to(run.device), edge_index.to(run.device))
    output = output.cpu().detach().numpy()
    if SCALE_MODE == "minmax":
        output = output * (run.scaling[4] - run.scaling[3]) + run.scaling[3]
    elif SCALE_MODE == "standard":
        output = output * run.scaling[4] + run.scaling[3]
    print(f"Predicted aerodynamic force: {run.aero_forces_out[sample]}")
    # Visualize pointclouds from the database and the prediction
    run.visualize_pointcloud(points, values, window_name="database sample")
    run.visualize_pointcloud(points, output[:, 0], window_name="GNN prediction")


if __name__ == "__main__":
    main()
