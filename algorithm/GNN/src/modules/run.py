import numpy as np
import pandas as pd
import torch
import wandb
import open3d as o3d
import matplotlib.cm as cm
import torch_geometric.transforms as T
from matplotlib.colors import Normalize
import copy

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
        self.train_loss = history.train_loss.values
        self.val_loss = history.val_loss.values
        # Get model
        model_artifact = api.artifact(project_path + "/model:" + run_name)
        self.model = torch.jit.load(
            model_artifact.download() + r"/scripted_model_best.pt"
        ).to(self.device)
        self.model.eval()
        # Get scaling parameters
        scale_artifact = api.artifact(project_path + "/scaling-parameters:" + run_name)
        scaling_dict = np.load(
            scale_artifact.download() + "/scaling.npy", allow_pickle=True
        ).all()
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
        self.dataset = datafile["database"]
        self.pitch_angles = datafile["pitch_angles"]
        self.yaw_angles = datafile["yaw_angles"]
        print("Dataset loaded.")

    def transform_dataset(self):
        # Transform the dataset to PyTorch Geometric Data objects
        transformed_dataset = []
        GCNNorm = T.GCNNorm(add_self_loops=True)
        for graph in self.dataset:
            # Create a PyTorch Geometric Data object
            data = GCNNorm(graph)
            # Add the graph to the transformed dataset
            transformed_dataset.append(data)
        self.dataset = transformed_dataset

    def scale_dataset(self):
        scaling = self.scaling
        # Deep copy each Data object to avoid shared references
        scaled_dataset = [copy.deepcopy(graph) for graph in self.dataset]
        for graph in scaled_dataset:
            graph.x[:, Const.vel_idx] /= scaling[0]
            if Const.scale_mode == "minmax":
                graph.x[:, Const.pos_idx] = (graph.x[:, Const.pos_idx] - scaling[1]) / (
                    scaling[2] - scaling[1]
                )
                graph.y[:, Const.flow_idx] = (
                    graph.y[:, Const.flow_idx] - scaling[3]
                ) / (scaling[4] - scaling[3])
            elif Const.scale_mode == "standard":
                graph.x[:, Const.pos_idx] = (
                    graph.x[:, Const.pos_idx] - scaling[1].astype(np.float32)
                ) / scaling[2].astype(np.float32)
                graph.y[:, Const.flow_idx] = (
                    graph.y[:, Const.flow_idx] - scaling[3].astype(np.float32)
                ) / scaling[4].astype(np.float32)
        self.scaled_dataset = scaled_dataset
        print("Dataset scaled.")

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

    def compute_aerodynamic_forces(self, wind_speed: float, scale_mode: str = "minmax"):
        self.aero_force_errors = np.zeros((len(self.dataset), 3))
        self.aero_forces_in = np.zeros((len(self.dataset), 3))
        self.aero_forces_out = np.zeros((len(self.dataset), 3))
        for i, zip_data in enumerate(zip(self.dataset, self.scaled_dataset)):
            data, scaled_data = zip_data
            # Get input: v_x, v_y, v_z, x, y, z
            if Const.in_dim == 6:
                x = scaled_data.x[:, Const.vel_idx + Const.pos_idx]
            elif Const.in_dim == 9:
                x = scaled_data.x[
                    :, Const.vel_idx + Const.pos_idx + Const.face_normal_idx
                ]
            # Compute forward pass
            self.model.to(self.device)
            self.model.eval()
            output = (
                self.model(x.to(self.device), data.edge_index.to(self.device))
                .detach()
                .cpu()
                .numpy()
            )
            # Rescale output
            if scale_mode == "minmax":
                output = output * (self.scaling[4] - self.scaling[3]) + self.scaling[3]
            elif scale_mode == "standard":
                output = output * self.scaling[4] + self.scaling[3]
            press_coeff = output[:, 0]
            fric_coeff = output[:, 1:]
            # Compute predicted centroidal aerodynamic force
            dyn_press = 0.5 * 1.225 * wind_speed**2
            face_normals = data.x[:, Const.face_normal_idx].numpy()
            areas = data.x[:, Const.area_idx].numpy()
            pressure = press_coeff.reshape(-1, 1) * dyn_press
            friction = fric_coeff * dyn_press
            d_force = pressure * face_normals * areas + friction * areas
            pred_body_force = np.sum(d_force, axis=0)
            # Compute dataset aerodynamic force
            pressure = data.y[:, Const.flow_idx[0]].numpy().reshape(-1, 1) * dyn_press
            friction = data.y[:, Const.flow_idx[1:]].numpy() * dyn_press
            d_force = pressure * face_normals * areas + friction * areas
            dataset_body_force = np.sum(d_force, axis=0)
            # rotate forces to aerodynamic frame
            pitch_angle = self.pitch_angles[i]
            yaw_angle = self.yaw_angles[i]
            self.robot.set_state(pitch_angle, yaw_angle, np.zeros(self.robot.nDOF))
            world_H_base = self.robot.compute_world_H_link("root_link")
            pred_aero_force = np.dot(world_H_base[:3, :3], pred_body_force)
            dataset_aero_force = np.dot(world_H_base[:3, :3], dataset_body_force)
            # Save data
            self.aero_forces_in[i, :] = dataset_aero_force
            self.aero_forces_out[i, :] = pred_aero_force
            self.aero_force_errors[i, :] = np.abs(dataset_aero_force - pred_aero_force)
        # Compute Mean Squared Error
        aero_force_rmse = np.sqrt(np.mean(self.aero_force_errors**2, axis=0))
        # Display MSE
        print(f"RMSE Drag Force: {aero_force_rmse[2]}")
        print(f"RMSE Lift Force: {aero_force_rmse[1]}")
        print(f"RMSE Side Force: {aero_force_rmse[0]}")
