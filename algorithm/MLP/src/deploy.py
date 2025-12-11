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
from resolve_robotics_uri_py import resolve_robotics_uri

from modules.constants import Const
from modules.run import Run
from modules.robot import Robot

RUN_NAME = "trial-17"
SCALE_MODE = "standard"  # "standard", "minmax"
PITCH = [0.0, 40.0, 90.0]
YAW = [0.0]
WIND_SPEED = 17.0


def main():
    # Get root directory path
    root = Path(__file__).parents[1]

    # Set the database path
    dataset_dir = root / "test_cases" / "ironcub" / "database"
    dataset_file = dataset_dir / "ironcub-hovering-bodyframe.npz"

    # Create output directory
    output_dir = root / "output" / RUN_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize robot object
    robot_name = "iRonCub-Mk3"
    urdf_path = str(resolve_robotics_uri("package://iRonCub-Mk3/model.urdf"))
    robot = Robot(robot_name, urdf_path)

    # Initialize the run object
    run = Run(robot)
    run.load_train_from_wandb("ami-iit/MLP-iRonAero", RUN_NAME)
    # run.plot_losses()
    run.load_dataset(dataset_file)

    # Get model input layer dimension (MLP vs MLPN)
    Const.in_dim = run.model.input_layer.weight.shape[1]

    # # Compute aerodynamic forces MSE
    run.compute_aerodynamic_forces(WIND_SPEED, SCALE_MODE, data_split="train")
    run.compute_aerodynamic_forces(WIND_SPEED, SCALE_MODE, data_split="validation")
    run.compute_aerodynamic_forces(WIND_SPEED, SCALE_MODE, data_split="all")

    # # Compute custom output
    # # denser_pitches = [i for i in range(-170, 180, 10)]
    # # aero_forces_out = run.compute_aerodynamic_forces_custom_input(
    # #     WIND_SPEED,
    # #     SCALE_MODE,
    # #     pitch_angles=denser_pitches,
    # #     yaw_angles=[0.0],
    # # )

    ### "polar" curves
    # positive pitch
    ids = np.where(run.yaw_angles == 0)[0]  # already sorted by dataset construction
    train_ids = np.array([i for i in ids if i in run.train_ids])
    val_ids = np.array([i for i in ids if i in run.val_ids])
    pitches = run.pitch_angles[ids]
    force_out = run.aero_forces_out[ids]
    train_pitches = run.pitch_angles[train_ids]
    train_force_in = run.aero_forces_in[train_ids]
    val_pitches = run.pitch_angles[val_ids]
    val_force_in = run.aero_forces_in[val_ids]
    # negative pitch
    ids = np.where(run.yaw_angles == 180)[0][:-1]
    train_ids = np.array([i for i in ids if i in run.train_ids])
    val_ids = np.array([i for i in ids if i in run.val_ids])
    neg_pitches = run.pitch_angles[ids] - 180
    pitches = np.concatenate((neg_pitches, pitches), axis=0)
    neg_force_out = run.aero_forces_out[ids]
    neg_force_out[:, [0, 1]] = -neg_force_out[:, [0, 1]]
    force_out = np.concatenate((neg_force_out, force_out), axis=0)
    neg_train_pitches = run.pitch_angles[train_ids] - 180
    train_pitches = np.concatenate((neg_train_pitches, train_pitches), axis=0)
    neg_train_force_in = run.aero_forces_in[train_ids]
    neg_train_force_in[:, [0, 1]] = -neg_train_force_in[:, [0, 1]]
    train_force_in = np.concatenate((neg_train_force_in, train_force_in), axis=0)
    neg_val_pitches = run.pitch_angles[val_ids] - 180
    val_pitches = np.concatenate((neg_val_pitches, val_pitches), axis=0)
    neg_val_force_in = run.aero_forces_in[val_ids]
    neg_val_force_in[:, [0, 1]] = -neg_val_force_in[:, [0, 1]]
    val_force_in = np.concatenate((neg_val_force_in, val_force_in), axis=0)
    force_out[:, [0, 2]] = -force_out[:, [0, 2]]
    train_force_in[:, [0, 2]] = -train_force_in[:, [0, 2]]
    val_force_in[:, [0, 2]] = -val_force_in[:, [0, 2]]
    out_data = {
        "pitch_angles": pitches,
        "train_pitch_angles": train_pitches,
        "val_pitch_angles": val_pitches,
        "aero_forces_out": force_out,
        "train_aero_forces_in": train_force_in,
        "val_aero_forces_in": val_force_in,
    }
    np.savez(output_dir / "data.npz", **out_data)

    # 3D Visualization
    for yaw in YAW:
        for pitch in PITCH:
            print(f"Pitch: {pitch}, Yaw: {yaw}")
            idx = np.where((run.pitch_angles == pitch) & (run.yaw_angles == yaw))[0][0]
            set_split = "train" if idx in run.train_ids else "validation"
            print(f"Sample {idx} from {set_split} set")
            points = run.dataset[idx][:, Const.pos_idx]
            values = run.dataset[idx][:, Const.flow_idx[0]]
            print(f"Database aerodynamic force: {run.aero_forces_in[idx]}")
            # Compute prediction with the model using scaled input
            input_vel = run.dataset[idx][:, Const.vel_idx] / run.scaling[0]
            if SCALE_MODE == "minmax":
                input_pos = (points - run.scaling[1]) / (
                    run.scaling[2] - run.scaling[1]
                )
            elif SCALE_MODE == "standard":
                input_pos = (points - run.scaling[1]) / run.scaling[2]
            input = np.concatenate((input_vel, input_pos), axis=1)
            if Const.in_dim == 9:  # MLP with face normals
                input_n = run.dataset[idx][:, Const.face_normal_idx]
                input = np.concatenate((input, input_n), axis=1)
            input = torch.from_numpy(input).float().to(run.device)
            out = run.model(input)
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
            run.visualize_pointcloud(w_points, out[:, 0], window_name="MLP prediction")


if __name__ == "__main__":
    main()
