import numpy as np
import pandas as pd
import torch
import wandb
import open3d as o3d
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from modules.constants import Const


class Run:
    def __init__(self, robot):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.robot = robot

    def load_train_from_wandb(self, project_path: str, run_name: str):
        print(f"Loading model from: {project_path}/{run_name}")
        # Get run data
        api = wandb.Api()
        runs = wandb.Api().runs(project_path, order="-created_at")
        run = [runs[i] for i in range(len(runs)) if runs[i].name == run_name][0]
        self.config = run.config
        # Get losses history
        history = run.history()
        self.epoch = history.iteration.values
        self.train_loss = history.train_loss.values
        self.val_loss = history.val_loss.values
        # Get model
        model_artifact = api.artifact(project_path + "/model:" + run_name)
        self.model = torch.jit.load(
            model_artifact.download() + r"/scripted_model.pt"
        ).to(self.device)
        self.model.eval()
        # Get scaling parameters
        scale_artifact = api.artifact(project_path + "/scaling-parameters:" + run_name)
        scaling_dict = np.load(
            scale_artifact.download() + "/scaling.npy", allow_pickle=True
        ).item()
        self.scaling = [scaling_dict[idx] for idx in scaling_dict.keys()]

        # Get dataset indices
        dataset_indices_artifact = api.artifact(
            project_path + "/dataset-indices:" + run_name
        ).get("df")
        df = pd.DataFrame(
            data=dataset_indices_artifact.data, columns=dataset_indices_artifact.columns
        )
        train_ids = pd.DataFrame(df, columns=["Training set"]).values.reshape(-1)
        val_ids = pd.DataFrame(df, columns=["Validation set"]).values.reshape(-1)
        self.train_ids = [train_id for train_id in train_ids if train_id is not None]
        self.val_ids = [val_id for val_id in val_ids if val_id is not None]
        print("Model loaded.")

    def load_local_model(self, model_path: str):
        # Get model
        self.model = torch.jit.load(model_path / "scripted_model.pt").to(self.device)
        self.model.eval()
        # Get scaling parameters
        scaling_dict = np.load(model_path / "scaling.npy", allow_pickle=True).all()
        self.scaling = [scaling_dict[idx] for idx in scaling_dict.keys()]

    def load_dataset(self, datafile_path: str):
        print(f"Loading dataset from: {datafile_path}")
        datafile = np.load(datafile_path, allow_pickle=True)
        self.dataset = datafile["data"].tolist()["data"]
        self.pitch_angles = np.array(datafile["data"].tolist()["pitch_angles"])
        self.yaw_angles = np.array(datafile["data"].tolist()["yaw_angles"])
        print("Dataset loaded.")

    def scale_dataset_minmax(self):
        # WARNING: Don't use this method (referencing problems can occur!)
        dtbs = self.dataset
        for sample in range(len(self.dataset)):
            dtbs[sample][:, Const.vel_idx] /= self.scaling[0]
            dtbs[sample][:, Const.pos_idx] = (
                dtbs[sample][:, Const.pos_idx] - self.scaling[1]
            ) / (self.scaling[2] - self.scaling[1])
            dtbs[sample][:, Const.flow_idx] = (
                dtbs[sample][:, Const.flow_idx] - self.scaling[3]
            ) / (self.scaling[4] - self.scaling[3])
        self.scaled_database = dtbs
        print("Dataset scaled.")

    def plot_losses(self):
        yticks = np.hstack((np.array([0.08, 0.09]), np.arange(0.1, 1.1, 0.1)))
        plt.figure(figsize=(12, 8))
        plt.semilogy(
            self.epoch,
            self.train_loss,
            linewidth=2,
            label="Training Loss",
            color="blue",
        )
        plt.semilogy(
            self.epoch,
            self.val_loss,
            linewidth=2,
            label="Validation Loss",
            color="orange",
        )
        plt.xlabel("Epochs", fontsize=24)
        plt.ylabel("Loss", fontsize=24)
        plt.xlim([0, 5000])
        plt.ylim([0.08, 1.0])
        plt.xticks(fontsize=18)
        plt.yticks(ticks=yticks, labels=np.round(yticks, 2), fontsize=18)
        plt.legend(fontsize=24)
        plt.grid()
        plt.show()

    def visualize_pointcloud(self, points, values, window_name="Open3D"):
        # Normalize the colormap
        norm = Normalize(vmin=-2, vmax=1)
        normalized_flow_variable = norm(values)
        colormap = cm.jet
        colors = colormap(normalized_flow_variable)[:, :3]
        # Create a point cloud from the dataset
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # Create a coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0, 0, 0]
        )
        # Set visualization parameters
        zoom = 0.75
        x_cen = (points[:, 0].max() + points[:, 0].min()) / 2
        y_cen = (points[:, 1].max() + points[:, 1].min()) / 2
        z_cen = (points[:, 2].max() + points[:, 2].min()) / 2
        center = [x_cen, y_cen, z_cen]
        front = [1.0, 0.0, 1.0]
        up = [0.0, 1.0, 0.0]
        # Display the pointcloud
        o3d.visualization.draw_geometries(
            [frame, pcd],
            zoom=zoom,
            lookat=center,
            front=front,
            up=up,
            window_name=window_name,
        )

    def compute_aerodynamic_forces(
        self,
        wind_speed: float,
        scale_mode: str = "standard",
        data_split: str = "train",
    ):
        dyn_press = 0.5 * 1.225 * wind_speed**2
        self.aero_forces_in = np.empty((0, 3))
        self.aero_forces_out = np.empty((0, 3))
        self.aero_force_errors = np.empty((0, 3))
        self.aero_force_norm_errs = np.empty((0))
        if data_split == "train":
            samples = [int(i) for i in self.train_ids if i is not pd.isna(i)]
        elif data_split == "validation":
            samples = [int(i) for i in self.val_ids if not pd.isna(i)]
        elif data_split == "all":
            samples = range(len(self.dataset))
        for sample in samples:
            sim = self.dataset[int(sample)]
            pitch_angle = self.pitch_angles[int(sample)]
            yaw_angle = self.yaw_angles[int(sample)]
            # Get input: v_x, v_y, v_z, x, y, z
            if Const.in_dim == 6:  # MLP
                input = sim[:, Const.vel_idx + Const.pos_idx]
            elif Const.in_dim == 9:  # MLPN
                input = sim[:, Const.vel_idx + Const.pos_idx + Const.face_normal_idx]
            input[:, :3] /= self.scaling[0]
            if scale_mode == "minmax":
                input[:, 3:6] = (input[:, 3:6] - self.scaling[1]) / (
                    self.scaling[2] - self.scaling[1]
                )
            elif scale_mode == "standard":
                input[:, 3:6] = (input[:, 3:6] - self.scaling[1]) / self.scaling[2]
            input = torch.tensor(input, dtype=torch.float32).to(self.device)
            # Compute forward pass
            self.model.to(self.device)
            self.model.eval()
            output = self.model(input).detach().cpu().numpy()
            if scale_mode == "minmax":
                output = output * (self.scaling[4] - self.scaling[3]) + self.scaling[3]
            elif scale_mode == "standard":
                output = output * self.scaling[4] + self.scaling[3]
            press_coeff = output[:, 0]
            fric_coeff = output[:, 1:]
            # Compute predicted centroidal aerodynamic force
            face_normals = sim[:, Const.face_normal_idx]
            areas = sim[:, Const.area_idx]
            pressure = press_coeff.reshape(-1, 1) * dyn_press
            friction = fric_coeff * dyn_press
            d_force = pressure * face_normals * areas + friction * areas
            pred_body_force = np.sum(d_force, axis=0)
            # Compute dataset aerodynamic force
            pressure = sim[:, Const.flow_idx[0]].reshape(-1, 1) * dyn_press
            friction = sim[:, Const.flow_idx[1:]] * dyn_press
            d_force = pressure * face_normals * areas + friction * areas
            dataset_body_force = np.sum(d_force, axis=0)
            # rotate forces to aerodynamic frame
            self.robot.set_state(pitch_angle, yaw_angle, np.zeros(self.robot.nDOF))
            world_H_base = self.robot.compute_world_H_link("root_link")
            pred_aero_force = np.dot(world_H_base[:3, :3], pred_body_force)
            dataset_aero_force = np.dot(world_H_base[:3, :3], dataset_body_force)
            aero_force_error = np.abs(dataset_aero_force - pred_aero_force)
            aero_force_norm_error = np.abs(
                np.linalg.norm(dataset_aero_force) - np.linalg.norm(pred_aero_force)
            ) / np.linalg.norm(dataset_aero_force)
            # Save data
            self.aero_forces_in = np.append(
                self.aero_forces_in, [dataset_aero_force], axis=0
            )
            self.aero_forces_out = np.append(
                self.aero_forces_out, [pred_aero_force], axis=0
            )
            self.aero_force_errors = np.append(
                self.aero_force_errors, [aero_force_error], axis=0
            )
            self.aero_force_norm_errs = np.append(
                self.aero_force_norm_errs, [aero_force_norm_error]
            )
        # Compute Root Mean Squared Errors
        aero_force_rmse = np.sqrt(np.mean(self.aero_force_errors**2, axis=0))
        aero_force_norm_rmse = np.sqrt(np.mean(self.aero_force_norm_errs**2)) * 100
        # Display RMSE
        print(f"RMSE Drag Force ({data_split}): {aero_force_rmse[2]}")
        print(f"RMSE Lift Force ({data_split}): {aero_force_rmse[1]}")
        print(f"RMSE Side Force ({data_split}): {aero_force_rmse[0]}")
        print(f"%RMSE Norm Force ({data_split}): {aero_force_norm_rmse}")
