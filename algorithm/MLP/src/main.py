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

from modules import preprocess as pre
from modules import models as mod
from modules import train as train
from modules import output as out
from modules import log
from modules.constants import Const


def main():
    # Get default values
    default_values = Const.get_default_values()
    # Read configuration file
    print("Reading config file")
    if len(sys.argv) < 2:
        print("\n\033[31mNo .cfg file provided in input.\nKilling execution \033[0m")
        sys.exit()
    Const.config_path = str(sys.argv[1])
    config_options = pre.read_config_file(Const.config_path)

    # Set constant values from options dictionary
    Const.set_val_from_options(config_options)

    # Init wandb logging
    if Const.wandb_logging:
        print("Wandb logging enabled")
        Const.run_name = log.init_wandb_project(options=config_options)

    # Print options for user check
    pre.print_options(config_options, default_values)

    # Load dataset
    dataset, _, _ = pre.load_dataset()

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    # Set random seed
    pre.set_seed(Const.rnd_seed)
    print(f"Random seed set as {Const.rnd_seed}")

    # Split dataset
    data_train_list, data_val_list, data_test_list, indices = pre.split_dataset(
        dataset, Const.val_set, Const.test_set
    )
    # train_idx = indices[: len(data_train_list)]
    # val_idx = indices[len(data_train_list) : len(data_train_list) + len(data_val_list)]
    # test_idx = indices[len(data_train_list) + len(data_val_list) :]

    # Concatenate sample points
    data_train = np.concatenate(data_train_list, axis=0)
    data_val = np.concatenate(data_val_list, axis=0)
    full_dataset = np.concatenate((data_train, data_val), axis=0)
    print(f"Training set size: {data_train.shape}")
    print(f"Validation set size: {data_val.shape}")
    if data_test_list:
        data_test = np.concatenate(data_test_list, axis=0)
        full_dataset = np.concatenate((full_dataset, data_test), axis=0)
        print(f"Testing set size: {data_test.shape}")

    # Scale dataset
    print("Scaling dataset")
    scaling = pre.compute_scaling(full_dataset)
    data_train = pre.scale_dataset(data_train, scaling)
    data_val = pre.scale_dataset(data_val, scaling)
    if data_test_list:
        data_test = pre.scale_dataset(data_test, scaling)

    # Create dataloaders
    if Const.mode == "mlp" or Const.mode == "mlp-tuning":
        input_indices = Const.vel_idx + Const.pos_idx
    elif Const.mode == "mlpn":
        input_indices = Const.vel_idx + Const.pos_idx + Const.face_normal_idx
    train_dataset = mod.MlpDataset(
        data_train[:, input_indices],
        data_train[:, Const.flow_idx],
        Const.batch_size,
    )
    val_dataset = mod.MlpDataset(
        data_val[:, input_indices],
        data_val[:, Const.flow_idx],
        Const.batch_size,
    )
    train_dl = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False)

    if Const.mode == "mlp" or Const.mode == "mlpn":
        # Define the MLP model
        model = mod.MLP().to(device)

        # Initialize weights
        mod.initialize_weights_xavier_normal(model)
        # Define loss function
        loss = torch.nn.MSELoss()
        # Define optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=Const.initial_lr, weight_decay=Const.reg_par
        )

        # Print model summary
        print("Model summary")
        torchsummary.summary(model.to(device), (Const.in_dim,))

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
        Const.out_dir = out.gen_folder(Const.out_dir)
        # Save the scaling values
        out.save_scaling(scaling)
        # save the trained and best encoder
        out.save_model(
            model,
            optimizer,
            torch.ones((1, Const.in_dim)),
            best_model,
        )
        # save the history file
        out.write_hystory(history)
        # Save the indices of the the sub-sets into an xlsx file
        out.write_datasets(indices, len(data_train_list), len(data_val_list))

    elif Const.mode == "mlp-tuning":

        Const.optuna_trial = 0

        # Define the objective function
        def objective(trial):
            print(f"Optuna trial: {Const.optuna_trial}/{Const.n_trials}")
            Const.batch_size = trial.suggest_int("batch_size", 50000, 500000)
            Const.initial_lr = trial.suggest_float("initial_lr", 1e-4, 1e-2)
            Const.reg_par = trial.suggest_float("reg_par", 1e-9, 1e-5)
            Const.hid_layers = trial.suggest_int("hid_layers", 4, 12)
            hid_dim_pow = trial.suggest_int("hid_dim_pow", 7, 9)
            Const.hid_dim = int(2**hid_dim_pow)
            # Define the MLP model
            model = mod.MLP().to(device)
            # Initialize weights
            mod.initialize_weights_xavier_normal(model)
            # Define loss function
            loss = torch.nn.MSELoss()
            # Define optimizer
            optimizer = torch.optim.Adam(
                model.parameters(), lr=Const.initial_lr, weight_decay=Const.reg_par
            )
            # Print model summary
            print("Model summary")
            torchsummary.summary(model.to(device), (Const.in_dim,))
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
            history, model, best_model = train.train_MLP(
                train_dl,
                val_dl,
                model,
                loss,
                optimizer,
                device,
            )
            # Compute optuna score
            train_loss = history[-1][1]
            val_loss = history[-1][2]
            if Const.wandb_logging:
                wandb.log(
                    {
                        "lr": Const.initial_lr,
                        "reg_par": Const.reg_par,
                        "hid_layers": Const.hid_layers,
                        "hid_dim": Const.hid_dim,
                        "pareto_train_loss": train_loss,
                        "pareto_val_loss": val_loss,
                    }
                )
            Const.optuna_trial += 1
            return val_loss

        # Define the study
        study = optuna.create_study(directions=["minimize"])
        study.optimize(objective, n_trials=Const.n_trials)

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
        sys.exit("\nERROR: " + Const.mode + " mode not existing.\nTerminating!\n")

    # WANDB LOGGING
    if Const.wandb_logging and Const.mode != "mlp-tuning":
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


if __name__ == "__main__":
    main()
