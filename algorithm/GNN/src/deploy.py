"""
Author: Antonello Paolino
Date: 2025-07-01
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

RUN_NAME = "trial-10"
SCALE_MODE = "standard"  # "standard", "minmax"
PITCH = [0.0, 40.0, 90.0]
YAW = [0.0]
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
    run.load_train_from_wandb("ami-iit/GNN-iRonAero", RUN_NAME)
    # run.plot_losses()
    run.load_dataset(dataset_file)
    run.transform_dataset()
    run.scale_dataset()

    # Set model input layer dimension (MLP vs MLPN)
    Const.in_dim = 9

    # Compute aerodynamic forces MSE
    run.compute_aerodynamic_forces(WIND_SPEED, SCALE_MODE)

    # 3D Visualization
    for yaw in YAW:
        for pitch in PITCH:
            print(f"Pitch: {pitch}, Yaw: {yaw}")
            idx = np.where((run.pitch_angles == pitch) & (run.yaw_angles == yaw))[0][0]
            set_split = "train" if idx in run.train_ids else "validation"
            print(f"Sample {idx} from {set_split} set")
            points = run.dataset[idx].x[:, Const.pos_idx].numpy()
            values = run.dataset[idx].y[:, Const.flow_idx[0]].numpy()
            print(f"Database aerodynamic force: {run.aero_forces_in[idx]}")
            # Compute prediction with the model using scaled input
            x = run.scaled_dataset[idx].x[
                :, Const.vel_idx + Const.pos_idx + Const.face_normal_idx
            ]
            edge_index = run.scaled_dataset[idx].edge_index
            out = run.model(x.to(run.device), edge_index.to(run.device))
            out = out.cpu().detach().numpy()
            if SCALE_MODE == "minmax":
                out = out * (run.scaling[4] - run.scaling[3]) + run.scaling[3]
            elif SCALE_MODE == "standard":
                out = out * run.scaling[4] + run.scaling[3]
            print(f"Predicted aerodynamic force: {run.aero_forces_out[idx]}")
            # Visualize pointclouds from the database and the prediction
            run.robot.set_state(pitch, yaw, np.zeros(run.robot.nDOF))
            world_H_base = run.robot.compute_world_H_link("root_link")
            w_points = (world_H_base[:3, :3] @ points.T + world_H_base[:3, 3:4]).T
            run.visualize_pointcloud(w_points, values, window_name="database sample")
            run.visualize_pointcloud(w_points, out[:, 0], window_name="GNN prediction")


if __name__ == "__main__":
    main()
