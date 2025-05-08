"""
Author: Antonello Paolino
Date: 2025-01-31
Description:    Main module to execute the training of the Multi-Layer
                Perceptron for Aerodynamics prediction of iRonCub surface
                flow variables (MLP-Aero).
"""

import numpy as np
import sys
import torch
import torch.nn as nn
import torch.onnx
import torch.jit
from torch.utils.data import DataLoader
import random as random
import time as time
import torchsummary
import optuna
from optuna.trial import TrialState
import wandb
from pathlib import Path

from modules import preprocess as pre
from modules import models as mod
from modules import train as train
from modules import output as out
from modules import log
from modules.constants import Const

from modules.run import Run

RUN_NAME = "trial-3"
PITCH = 90.0
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
    run.load_train_from_wandb("ami-iit/MLP-iRonAero", RUN_NAME)
    run.load_dataset(dataset_file)

    # Compute aerodynamic forces MSE
    run.compute_aerodynamic_forces(WIND_SPEED)
    print(f"Database aerodynamic force: {run.aero_forces_in[sample]}")
    print(f"Predicted aerodynamic force: {run.aero_forces_out[sample]}")

    # Visualization
    # Compute dataset sample
    sample = np.where((run.pitch_angles == PITCH) & (run.yaw_angles == YAW))[0][0]
    points = run.dataset[sample][:, Const.pos_idx]
    values = run.dataset[sample][:, Const.flow_idx[0]]
    # Compute prediction with the model using scaled input
    input_vel = run.dataset[sample][:, Const.vel_idx] / run.scaling[0]
    input_pos = (points - run.scaling[1]) / (run.scaling[2] - run.scaling[1])
    input = np.concatenate((input_vel, input_pos), axis=1)
    input = torch.from_numpy(input).float().to(run.device)
    output = run.model(input)
    output = output.cpu().detach().numpy()
    output = output * (run.scaling[4] - run.scaling[3]) + run.scaling[3]
    # Visualize pointclouds from the database and the prediction
    run.visualize_pointcloud(points, values, window_name="database sample")
    run.visualize_pointcloud(points, output[:, 0], window_name="MLP prediction")


if __name__ == "__main__":
    main()
