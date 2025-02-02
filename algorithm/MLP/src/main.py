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
import random as random
import time as time
import torchsummary
from pathlib import Path

from modules import preprocess as pre
from modules import models as mod
from modules import train as train
from modules import output as out
from modules import glob
from modules import log


def main():
    # Read configuration file
    print("Reading config file")
    if len(sys.argv) < 2:
        print("\n\033[31mNo .cfg file provided in input.\nKilling execution \033[0m")
        sys.exit()
    glob.config_path = str(sys.argv[1])
    options, default_values, add_opt = pre.read_config_file(glob.config_path)

    # Init wandb logging
    if options["wandb_logging"].lower() == "yes":
        print("Wandb logging enabled")
        glob.wandb_logging = True
        glob.run_name = log.init_wandb_project(options={**options, **add_opt})

    # Print options for user check
    pre.print_options(options, default_values, add_opt)

    # Load dataset
    glob.dataset_path = options["dataset_path"]
    dataset, _, _ = pre.load_dataset()

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    # Set random seed
    glob.rnd_seed = int(options["rnd_seed"])
    pre.set_seed(glob.rnd_seed)
    print(f"Random seed set as {glob.rnd_seed}")

    # Split dataset
    glob.val_set = float(options["val_set"])
    glob.test_set = float(options["test_set"])
    data_train_list, data_val_list, data_test_list, indices = pre.split_dataset(
        dataset, glob.val_set, glob.test_set
    )
    # train_idx = indices[: len(data_train_list)]
    # val_idx = indices[len(data_train_list) : len(data_train_list) + len(data_val_list)]
    # test_idx = indices[len(data_train_list) + len(data_val_list) :]

    # Concatenate sample points
    data_train = np.concatenate(data_train_list, axis=0)
    data_val = np.concatenate(data_val_list, axis=0)
    full_dataset = np.concatenate((data_train, data_val), axis=0)
    if data_test_list:
        data_test = np.concatenate(data_test_list, axis=0)
        full_dataset = np.concatenate((full_dataset, data_test), axis=0)

    # Scale dataset
    print("Scaling dataset")
    scaling = pre.compute_scaling(full_dataset)
    data_train = pre.scale_dataset(data_train, scaling)
    data_val = pre.scale_dataset(data_val, scaling)
    if data_test_list:
        data_test = pre.scale_dataset(data_test, scaling)

    # Create dataloaders
    glob.batch_size = int(options["batch_size"])
    dl_train = torch.utils.data.DataLoader(
        data_train, batch_size=glob.batch_size, shuffle=False
    )
    dl_val = torch.utils.data.DataLoader(
        data_val, batch_size=glob.batch_size, shuffle=False
    )

    # Define code mode
    glob.mode = options["mode"].lower()

    if glob.mode == "mlp":
        # Define the MLP model
        glob.in_dim = int(options["in_dim"])
        glob.out_dim = int(options["out_dim"])
        glob.hid_layers = int(options["hid_layers"])
        glob.hid_dim = int(options["hid_dim"])
        glob.dropout = float(options["dropout"])
        model = mod.MLP().to(device)

        # Initialize weights
        init_weights = torch.empty(glob.in_dim, glob.hid_dim)
        nn.init.xavier_normal_(init_weights)
        # Define loss function
        loss = torch.nn.MSELoss()
        # Define optimizer
        glob.lr = float(options["lr"])
        glob.reg_par = float(options["reg_par"])
        optimizer = torch.optim.Adam(
            model.parameters(), lr=glob.lr, weight_decay=glob.reg_par
        )

        # Print model summary
        print("Model summary")
        torchsummary.summary(model.to(device), (glob.in_dim,))

        # Move model to device
        model.to(device)

        # Count the number of trainable parameters
        train_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        data_size = data_train.size
        if data_size < train_param:
            print(
                "Warning: the number of trainable parameters is greater than the dataset size"
            )

        # Training
        glob.epochs = int(options["epochs"])
        history, model, best_model = train.train_MLP(
            dl_train,
            dl_val,
            model,
            loss,
            optimizer,
            device,
        )

        # Generate output
        print("Generating output")
        # Generate output folder
        glob.out_dir = out.gen_folder(options["out_dir"])
        # Save the scaling values
        out.save_scaling(scaling)
        # save the trained and best encoder
        out.save_model(
            model,
            optimizer,
            torch.ones((1, glob.in_dim)),
            best_model,
        )
        # save the history file
        out.write_hystory(history)
        # Save the indices of the the sub-sets into an xlsx file
        out.write_datasets(indices, data_train.shape[0], data_val.shape[0])

    else:
        sys.exit("\nERROR: " + options["mode"] + " mode not existing.\nTerminating!\n")

    # WANDB LOGGING
    if glob.wandb_logging:
        # Log the aerodynamic forces error of the training set
        log.log_aerodynamic_forces_error(
            data_train_list,
            model,
            device,
            scaling,
            "training",
        )
        # Log the aerodynamic forces error of the validation set
        log.log_aerodynamic_forces_error(
            data_val_list,
            model,
            device,
            scaling,
            "validation",
        )
        # Log the aerodynamic forces error of the test set
        if data_test_list:
            log.log_aerodynamic_forces_error(
                data_test_list,
                model,
                device,
                scaling,
                "test",
            )
        # Close wandb logging
        log.wandb.finish()

    print("checkpoint")


if __name__ == "__main__":
    main()
