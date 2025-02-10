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

from modules import preprocess as pre
from modules import models as mod
from modules import train as train
from modules import output as out
from modules import globals as glvar
from modules import log

VEL_IDX = [0, 1, 2]
POS_IDX = [22, 23, 24]
FLOW_IDX = [28, 29, 30, 31]


def main():
    # Read configuration file
    print("Reading config file")
    if len(sys.argv) < 2:
        print("\n\033[31mNo .cfg file provided in input.\nKilling execution \033[0m")
        sys.exit()
    glvar.config_path = str(sys.argv[1])
    options, default_values, add_opt = pre.read_config_file(glvar.config_path)

    # Define code mode
    glvar.mode = options["mode"].lower()

    # Init wandb logging
    if options["wandb_logging"].lower() == "yes" and glvar.mode != "mlp-tuning":
        print("Wandb logging enabled")
        glvar.wandb_logging = True
        glvar.run_name = log.init_wandb_project(options={**options, **add_opt})

    # Print options for user check
    pre.print_options(options, default_values, add_opt)

    # Load dataset
    glvar.dataset_path = options["dataset_path"]
    dataset, _, _ = pre.load_dataset()

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    # Set random seed
    glvar.rnd_seed = int(options["rnd_seed"])
    pre.set_seed(glvar.rnd_seed)
    print(f"Random seed set as {glvar.rnd_seed}")

    # Split dataset
    glvar.val_set = float(options["val_set"])
    glvar.test_set = float(options["test_set"])
    data_train_list, data_val_list, data_test_list, indices = pre.split_dataset(
        dataset, glvar.val_set, glvar.test_set
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
    glvar.vel_idx = VEL_IDX
    glvar.pos_idx = POS_IDX
    glvar.flow_idx = FLOW_IDX
    scaling = pre.compute_scaling(full_dataset)
    data_train = pre.scale_dataset(data_train, scaling)
    data_val = pre.scale_dataset(data_val, scaling)
    if data_test_list:
        data_test = pre.scale_dataset(data_test, scaling)

    # Create dataloaders
    glvar.batch_size = int(options["batch_size"])
    train_dataset = mod.MlpDataset(
        data_train[:, glvar.vel_idx + glvar.pos_idx], data_train[:, glvar.flow_idx]
    )
    val_dataset = mod.MlpDataset(
        data_val[:, glvar.vel_idx + glvar.pos_idx], data_val[:, glvar.flow_idx]
    )
    train_dl = DataLoader(train_dataset, batch_size=glvar.batch_size, shuffle=False)
    val_dl = DataLoader(val_dataset, batch_size=glvar.batch_size, shuffle=False)

    if glvar.mode == "mlp":
        # Define the MLP model
        glvar.in_dim = int(options["in_dim"])
        glvar.out_dim = int(options["out_dim"])
        glvar.hid_layers = int(options["hid_layers"])
        glvar.hid_dim = int(options["hid_dim"])
        glvar.dropout = float(options["dropout"])
        model = mod.MLP().to(device)

        # Initialize weights
        init_weights = torch.empty(glvar.in_dim, glvar.hid_dim)
        nn.init.xavier_normal_(init_weights)
        # Define loss function
        loss = torch.nn.MSELoss()
        # Define optimizer
        glvar.lr = float(options["lr"])
        glvar.reg_par = float(options["reg_par"])
        optimizer = torch.optim.Adam(
            model.parameters(), lr=glvar.lr, weight_decay=glvar.reg_par
        )

        # Print model summary
        print("Model summary")
        torchsummary.summary(model.to(device), (glvar.in_dim,))

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
        glvar.epochs = int(options["epochs"])
        history, model, best_model = train.train_MLP(
            train_dl,
            val_dl,
            model,
            loss,
            optimizer,
            device,
        )

        # Generate output
        print("Generating output")
        # Generate output folder
        glvar.out_dir = out.gen_folder(options["out_dir"])
        # Save the scaling values
        out.save_scaling(scaling)
        # save the trained and best encoder
        out.save_model(
            model,
            optimizer,
            torch.ones((1, glvar.in_dim)),
            best_model,
        )
        # save the history file
        out.write_hystory(history)
        # Save the indices of the the sub-sets into an xlsx file
        out.write_datasets(indices, data_train.shape[0], data_val.shape[0])

    elif glvar.mode == "mlp-tuning":

        # Define the objective function
        def objective(trial):
            glvar.in_dim = int(options["in_dim"])
            glvar.out_dim = int(options["out_dim"])
            glvar.hid_layers = trial.suggest_int("hid_layers", 5, 9)
            glvar.hid_dim = trial.suggest_int("hid_dim", 256, 1024)
            glvar.dropout = trial.suggest_float("dropout", 0.0, 0.2)
            model = mod.MLP().to(device)
            # Initialize weights
            init_weights = torch.empty(glvar.in_dim, glvar.hid_dim)
            nn.init.xavier_normal_(init_weights)
            # Define loss function
            loss = torch.nn.MSELoss()
            # Define optimizer
            glvar.lr = trial.suggest_float("lr", 1e-4, 1e-2)
            glvar.reg_par = trial.suggest_float("reg_par", 1e-9, 1e-5)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=glvar.lr, weight_decay=glvar.reg_par
            )
            # Print model summary
            print("Model summary")
            torchsummary.summary(model.to(device), (glvar.in_dim,))
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
            glvar.epochs = int(options["epochs"])
            history, model, best_model = train.train_MLP(
                train_dl,
                val_dl,
                model,
                loss,
                optimizer,
                device,
            )
            # Generate output
            print("Generating output")
            # Generate output folder
            glvar.out_dir = out.gen_folder(options["out_dir"])
            # Save the scaling values
            out.save_scaling(scaling)
            # save the trained and best encoder
            out.save_model(
                model,
                optimizer,
                torch.ones((1, glvar.in_dim)),
                best_model,
            )
            # save the history file
            out.write_hystory(history)
            # Save the indices of the the sub-sets into an xlsx file
            out.write_datasets(indices, data_train.shape[0], data_val.shape[0])
            # Compute optuna score
            train_loss = history[-1][1]
            val_loss = history[-1][2]
            return val_loss

        # Define the study
        study = optuna.create_study(directions=["minimize"])
        glvar.n_trials = int(options["n_trials"])
        study.optimize(objective, n_trials=glvar.n_trials)

        # get the output
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    else:
        sys.exit("\nERROR: " + options["mode"] + " mode not existing.\nTerminating!\n")

    # WANDB LOGGING
    if glvar.wandb_logging:
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
