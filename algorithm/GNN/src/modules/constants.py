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
        self.scale_mode = "standard"
        self.rnd_seed = None
        self.compile_model = False
        self.val_set = 15
        self.test_set = 0
        self.batch_size = 1000
        self.epochs = 1000
        self.lr_scheduler = None
        self.initial_lr = 1e-3
        self.lr_iters = 10000
        self.lr_patience = 50
        self.reg_par = 1e-6
        self.vel_idx = [0, 1, 2]
        self.pos_idx = [3, 4, 5]
        self.face_normal_idx = [6, 7, 8]
        self.area_idx = [9]
        self.flow_idx = [0, 1, 2, 3]
        # model parameters
        self.in_dim = None
        self.out_dim = None
        self.enc_layers = 3
        self.enc_dim = 32
        self.latent_dim = 16
        self.gnc_layers = 3
        self.dec_layers = 3
        self.dec_dim = 32
        self.dropout = 0.0
        # optuna parameters
        self.optuna_trial = -1
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
