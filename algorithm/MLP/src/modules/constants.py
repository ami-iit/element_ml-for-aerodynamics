"""
Author: Antonello Paolino
Date: 2025-02-12
Description: Implementing a singleton class for storing the constant values.
"""


def singleton(cls):
    return cls()


@singleton
class Const:
    def __init__(self):
        # general settings
        self.config_path = None
        self.dataset_path = None
        self.out_dir = "Out"
        # wandb settings
        self.wandb_logging = False
        self.entity = None
        self.project = None
        self.run_name = None
        # training parameters
        self.mode = "mlp"
        self.rnd_seed = None
        self.val_set = 15
        self.test_set = 0
        self.batch_size = 1000
        self.epochs = 1000
        self.lr = 1e-3
        self.reg_par = 1e-6
        self.vel_idx = [0, 1, 2]
        self.pos_idx = [22, 23, 24]
        self.face_normal_idx = [25, 26, 27]
        self.flow_idx = [28, 29, 30, 31]
        self.area_idx = [32]
        # model parameters
        self.in_dim = 6
        self.out_dim = 4
        self.hid_layers = 5
        self.hid_dim = 256
        self.dropout = 0.0
        # optuna parameters
        self.n_trials = 10

    def get_default_values(self):
        default_values = {}
        for key, value in self.__dict__.items():
            default_values[key] = value
        return default_values

    def set_val_from_options(self, options):
        for key, value in options.items():
            setattr(Const, key.lower(), self.convert_value(value))

    def convert_value(self, value):
        if value.lower() in ("true", "false", "yes", "no"):
            return value.lower() == "true" or value.lower() == "yes"
        elif value.isdigit():
            return int(value)
        try:
            return float(value)
        except ValueError:
            return value  # Return as string if conversion fails
