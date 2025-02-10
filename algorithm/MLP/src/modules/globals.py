"""
Author: Antonello Paolino
Date: 2025-01-31
Description:    Module for global variables and settings
"""

# Settings
config_path = None
dataset_path = None
wandb_logging = False
run_name = None
out_dir = "Out"

# Training parameters
mode = "mlp"
rnd_seed = None
val_set = 15
test_set = 0
batch_size = 1000
epochs = 1000
lr = 1e-3
reg_par = 1e-6

vel_idx = [0, 1, 2]
pos_idx = [22, 23, 24]
flow_idx = [28, 29, 30, 31]

# Model parameters
in_dim = 6
out_dim = 4
hid_layers = 5
hid_dim = 256
dropout = 0.0

# optuna parameters
n_trials = 10
