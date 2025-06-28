"""
Author: Antonello Paolino
Date: 2025-01-31
Description:    Main module to execute the training of the Multi-Layer
                Perceptron for Aerodynamics prediction of iRonCub surface
                flow variables (MLP-Aero).
"""

import sys
import time as time
import random as random
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import summary
import wandb
import optuna
from optuna.trial import TrialState

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
    # if len(sys.argv) < 2:
    #     print("\n\033[31mNo .cfg file provided in input.\nKilling execution \033[0m")
    #     sys.exit()
    # Const.config_path = str(sys.argv[1])
    Const.config_path = str(
        r"C:\Users\apaolino\code\element_ml-for-aerodynamics\algorithm\GNN\test_cases\ironcub\input.cfg"
    )
    config_options = pre.read_config_file(Const.config_path)

    # Set constant values from options dictionary
    Const.set_val_from_options(config_options)

    # Init wandb logging
    if Const.wandb_logging:
        print("Wandb logging enabled")
        Const.run_name = log.init_wandb_project(options=config_options)

    # Print options for user check
    pre.print_options(config_options, default_values)

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    # Set random seed
    pre.set_seed(Const.rnd_seed)
    print(f"Random seed set as {Const.rnd_seed}")

    # Load dataset
    dataset, _, _ = pre.load_dataset()

    # Scale dataset
    print("Scaling dataset")
    scaling = pre.compute_scaling(dataset)
    scaled_dataset = pre.scale_dataset(dataset, scaling)

    # Select input variables
    in_idxs = Const.pos_idx + Const.vel_idx + Const.face_normal_idx
    for graph in scaled_dataset:
        graph.x = graph.x[:, Const.pos_idx + Const.vel_idx + Const.face_normal_idx]

    # Split dataset
    data_train, data_val, data_test, indices = pre.split_dataset(
        scaled_dataset, Const.val_set, Const.test_set
    )
    train_indices = indices[: len(data_train)]
    val_indices = indices[len(data_train) : len(data_train) + len(data_val)]
    test_indices = indices[len(data_train) + len(data_val) :] if data_test else None

    # Create dataloaders
    train_dl = DataLoader(data_train, batch_size=Const.batch_size, shuffle=False)
    val_dl = DataLoader(data_val, batch_size=Const.batch_size, shuffle=False)

    if Const.mode == "gnn":
        # Define the input and output dimensions
        Const.in_dim = len(in_idxs) if Const.in_dim is None else Const.in_dim
        Const.out_dim = len(Const.flow_idx) if Const.out_dim is None else Const.out_dim

        # Compute dataset sizes
        train_len = np.sum([g.x.shape[0] for g in data_train])
        print(f"Train set size: [{train_len},{Const.in_dim}]")
        val_len = np.sum([g.x.shape[0] for g in data_val])
        print(f"Validation set size: [{val_len},{Const.in_dim}]")
        if data_test:
            test_len = np.sum([g.x.shape[0] for g in data_test])
            print(f"Test set size: [{test_len},{Const.in_dim}]")

        # Initialize the GNN model and weights
        model = mod.GNN()
        mod.initialize_weights_xavier_normal(model)
        # Define loss function
        loss = torch.nn.MSELoss()
        # Define optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=Const.initial_lr, weight_decay=Const.reg_par
        )
        # Print model summary
        print("Model summary")
        print(summary(model, data_train[0].x, data_train[0].edge_index))

        # Count the number of trainable parameters
        train_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal number of trainable parameters: {train_param}")
        if train_len < train_param:
            print(
                "Warning: the number of trainable parameters is greater than the dataset size"
            )

        # Move model to device
        model.to(device)

        # Training
        history, model, best_model = train.train_GNN(
            train_dl, val_dl, model, loss, optimizer, device
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
            (data_train[0].x, data_train[0].edge_index),
            best_model,
        )
        # save the history file
        out.write_hystory(history)
        # Save the indices of the the sub-sets into an xlsx file
        out.write_datasets(indices, len(data_train), len(data_val))

    elif Const.mode == "mlp-tuning" or Const.mode == "mlpn-tuning":

        Const.in_dim = len(in_idxs) if Const.in_dim is None else Const.in_dim
        Const.out_dim = len(Const.flow_idx) if Const.out_dim is None else Const.out_dim

        # Compute dataset sizes
        train_len = np.sum([g.x.shape[0] for g in data_train])
        print(f"Train set size: [{train_len},{Const.in_dim}]")
        val_len = np.sum([g.x.shape[0] for g in data_val])
        print(f"Validation set size: [{val_len},{Const.in_dim}]")
        if data_test:
            test_len = np.sum([g.x.shape[0] for g in data_test])
            print(f"Test set size: [{test_len},{Const.in_dim}]")

        Const.optuna_trial = 0

        # Define the objective function
        def objective(trial):
            print(f"Optuna trial: {Const.optuna_trial}/{Const.n_trials}")
            Const.batch_size = trial.suggest_int("batch_size", 1, 10)
            Const.initial_lr = trial.suggest_float("initial_lr", 1e-4, 1e-2)
            Const.reg_par = trial.suggest_float("reg_par", 1e-9, 1e-5)
            Const.hid_layers = trial.suggest_int("hid_layers", 1, 2)
            hid_dim_pow = trial.suggest_int("hid_dim_pow", 4, 5)
            Const.hid_dim = int(2**hid_dim_pow)
            # Initialize the GNN model and weights
            model = mod.GNN()
            mod.initialize_weights_xavier_normal(model)
            # Define loss function
            loss = torch.nn.MSELoss()
            # Define optimizer
            optimizer = torch.optim.Adam(
                model.parameters(), lr=Const.initial_lr, weight_decay=Const.reg_par
            )
            # Print model summary
            print("Model summary")
            print(summary(model, data_train[0]))
            # Count the number of trainable parameters
            train_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\nTotal number of trainable parameters: {train_param}")
            if train_len < train_param:
                print(
                    "Warning: the number of trainable parameters is greater than the dataset size"
                )
            # Move model to device
            model.to(device)
            # Training
            history, model, best_model = train.train_GNN(
                train_dl, val_dl, model, loss, optimizer, device
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
            dataset[train_indices],
            model,
            device,
            scaling,
            "training",
        )
        # Log the aerodynamic forces error of the validation set
        log.log_aerodynamic_forces_error(
            dataset[val_indices],
            model,
            device,
            scaling,
            "validation",
        )
        # Log the aerodynamic forces error of the test set
        if data_test:
            log.log_aerodynamic_forces_error(
                dataset[test_indices],
                model,
                device,
                scaling,
                "test",
            )
        # Close wandb logging
        log.wandb.finish()


if __name__ == "__main__":
    main()
