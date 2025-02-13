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

from modules.constants import Const


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
    # Retrieve options from the 'DEFAULT' section as dictionary
    options = dict(config.items("DEFAULT"))
    return options


def print_options(options, default_values):
    print("Configuration options:")
    headers = ["Option Name", "Option Value", "Origin"]
    # Sort data alphabetically by the option name
    all_values = {**default_values, **options}
    data = sorted(
        [
            (key.upper(), value, check_if_default(key, options, default_values))
            for key, value in all_values.items()
        ],
        key=lambda x: x[0],
    )
    print(tabulate.tabulate(data, headers=headers, tablefmt="grid"))


def check_if_default(key, options, default_values):
    # Check if a key is user-defined or default assigned.
    if key in options:
        return "User-defined"
    elif key in default_values and not key in options:
        return "Default"


def load_dataset():
    print(f"Loading dataset: {Const.dataset_path}")
    datafile = np.load(Const.dataset_path, allow_pickle=True)
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
    max_wind_speed = np.max(np.abs(data[:, Const.vel_idx]))
    X_min = np.min(data[:, Const.pos_idx], axis=0)
    X_max = np.max(data[:, Const.pos_idx], axis=0)
    Y_min = np.min(data[:, Const.flow_idx], axis=0)
    Y_max = np.max(data[:, Const.flow_idx], axis=0)
    return max_wind_speed, X_min, X_max, Y_min, Y_max


def scale_dataset(data, scaling):
    data[:, Const.vel_idx] /= scaling[0]
    data[:, Const.pos_idx] = (data[:, Const.pos_idx] - scaling[1]) / (
        scaling[2] - scaling[1]
    )
    data[:, Const.flow_idx] = (data[:, Const.flow_idx] - scaling[3]) / (
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
