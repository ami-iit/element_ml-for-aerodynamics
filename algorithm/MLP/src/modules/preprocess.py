"""
Author: Antonello Paolino
Date: 2025-01-31
Description:    Module for processing the input data.
"""

import os
import numpy as np
import random
import torch
import sys
import configparser
import tabulate
from pathlib import Path

from modules import globals as glvar


def read_config_file(file_path):
    # Create a temporary section header
    config_content = "[DEFAULT]\n"
    if not Path(file_path).is_file():
        print(f"\nConfiguration file {file_path} does not exist.\n")
        sys.exit()
    with open(file_path, "r") as file:
        config_content += file.read()

    # Read options
    config = configparser.ConfigParser(comment_prefixes=("%",))
    config.read_string(config_content)

    # Retrieve options from the 'DEFAULT' section
    options = dict(config.items("DEFAULT"))

    # Check if 'DATABASE_PATH' option is present (the only necessary)
    if options.get("dataset_path") is None:
        print("\nDATASET_PATH is missing in the configuration file.\n")
        sys.exit()
    # Define default values for missing options
    default_values = {
        "dataset_path": None,
        "wandb_logging": "yes",
        "mode": "mlp",
        "in_dim": 6,
        "hid_layers": 3,
        "hid_dim": 512,
        "out_dim": 4,
        "dropout": 0.0,
        "rnd_seed": None,
        "epochs": 1000,
        "batch_size": 5000,
        "lr": 1e-3,
        "reg_par": 1e-6,
        "val_set": 15,
        "test_set": 0,
        "out_dir": "Out",
        "n_trials": 10,
    }
    # Find keys present in dict1 but not in dict2
    keys_only_in_options = set(options.keys()) - set(default_values.keys())
    # Create a new dictionary with the additional values from dict1
    additional_values_dict = {key: options[key] for key in keys_only_in_options}

    def_val = []
    # Check and set default values for missing options
    for option, default_value in default_values.items():
        if option not in config["DEFAULT"]:
            def_val.append(str(option))
            config.set("DEFAULT", option, str(default_value))
    # Retrieve options from the 'DEFAULT' section
    options = dict(config.items("DEFAULT"))
    # Remove the additional values from dict1
    for key in keys_only_in_options:
        del options[key]
    return options, def_val, additional_values_dict


def print_options(options, default_values, keys_only_in_options):
    print("Configuration options:")
    headers = ["Option Name", "Option Value", "Origin"]
    # Sort data alphabetically by the option name
    data = sorted(
        [
            (key.upper(), value, check_if_default(key, default_values))
            for key, value in options.items()
        ],
        key=lambda x: x[0],
    )
    print(tabulate.tabulate(data, headers=headers, tablefmt="grid"))
    print("\nInput options (found in the .cfg) that are not used:")
    data = sorted(
        [
            (key.upper(), value, check_if_default(key, default_values))
            for key, value in keys_only_in_options.items()
        ],
        key=lambda x: x[0],
    )
    print(tabulate.tabulate(data, headers=headers, tablefmt="grid"))


def check_if_default(key, default_values):
    # Check if a key is user-defined or default assigned.
    return "DEFAULT" if key in default_values else "User Defined"


def load_dataset():
    print(f"Loading dataset: {glvar.dataset_path}")
    datafile = np.load(glvar.dataset_path, allow_pickle=True)
    data = datafile["data"].tolist()
    dataset = data["data"]
    pitch_angles = data["pitch_angles"]
    yaw_angles = data["yaw_angles"]
    return dataset, pitch_angles, yaw_angles


def set_seed(seed: int = 42) -> None:
    # Convert the string to None
    np.random.seed(seed)
    random.seed(seed)
    if seed != None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)


def compute_scaling(data):
    max_wind_speed = np.max(np.abs(data[:, glvar.vel_idx]))
    X_min = np.min(data[:, glvar.pos_idx], axis=0)
    X_max = np.max(data[:, glvar.pos_idx], axis=0)
    Y_min = np.min(data[:, glvar.flow_idx], axis=0)
    Y_max = np.max(data[:, glvar.flow_idx], axis=0)
    return max_wind_speed, X_min, X_max, Y_min, Y_max


def scale_dataset(data, scaling):
    data[:, glvar.vel_idx] /= scaling[0]
    data[:, glvar.pos_idx] = (data[:, glvar.pos_idx] - scaling[1]) / (
        scaling[2] - scaling[1]
    )
    data[:, glvar.flow_idx] = (data[:, glvar.flow_idx] - scaling[3]) / (
        scaling[4] - scaling[3]
    )
    return data


def split_dataset(dataset, p_val, p_test):
    # Shuffle the dataset
    samples_num = len(dataset)
    indices = np.arange(samples_num)
    np.random.shuffle(indices)
    dataset = [dataset[i] for i in indices]

    # Split dataset into train, val, test
    val_size = max(1, int(samples_num * p_val))
    test_size = int(samples_num * p_test)
    split_index_val = samples_num - (val_size + test_size)
    split_index_test = split_index_val + val_size

    data_train = dataset[:split_index_val]
    data_val = (
        dataset[split_index_val:split_index_test]
        if float(p_test) > 0
        else dataset[split_index_val:]
    )
    data_test = dataset[split_index_test:] if float(p_test) > 0 else None

    return data_train, data_val, data_test, indices
