"""
Author: Antonello Paolino
Date: 2024-10-01
Description:    This code uses the iDynTree package to retrieve the robot status,
                then it generates a 2D representation fo the 3D pressure map on
                the robot component surfaces and saves the images in a dataset.
"""

import numpy as np
import pickle
from pathlib import Path
from resolve_robotics_uri_py import resolve_robotics_uri
import torch
from torch_geometric.data import Data

from src.robot import Robot
from src.flow import FlowImporter
import src.functions as fn

WIND_INTENSITY = 17.0  # Fixed simulation wind intensity


def main():
    # Initialize robot object
    robot_name = "iRonCub-Mk3"
    urdf_path = str(resolve_robotics_uri("package://iRonCub-Mk3/model.urdf"))
    mesh_robot = Robot(robot_name, urdf_path)
    robot = Robot(robot_name, urdf_path)
    # Initialize flow object
    flow = FlowImporter()
    # Create a dataset output directory if not existing
    dataset_dir = Path(__file__).parents[0] / "disconnected-graphs-dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # Define graph directory
    graph_dir = Path(__file__).parents[0] / "disconnected-graphs"
    # Get the path to the raw data
    data_dir = input("Enter the path to the fluent data directory: ")
    data_path = Path(str(data_dir).strip())
    file_names = [file.name for file in data_path.rglob("*.dtbs") if file.is_file()]
    config_names = sorted(
        list(set([file_name.split("-")[0] for file_name in file_names]))
    )
    config_file_path = list(data_path.rglob("joint-configurations.csv"))[0]
    joint_configs = np.genfromtxt(config_file_path, delimiter=",", dtype=str)

    dataset = {}
    for config in config_names:
        joint_pos = joint_configs[joint_configs[:, 0] == config][0, 1:].astype(float)
        joint_pos *= np.pi / 180

        # Import and set mesh graph data
        graph_file = list(graph_dir.rglob(f"{config}-dual-graph.npy"))[0]
        graph_data = np.load(graph_file, allow_pickle=True).item()
        mesh_robot.set_state(0, 0, joint_pos)
        mesh_l_H_w = mesh_robot.compute_all_link_H_world()
        flow.import_mesh_graph(graph_data, robot.surface_list, mesh_l_H_w)
        pitch_yaw_angles = fn.find_pitch_yaw_angles(config, file_names)

        # Iterate over all pitch and yaw angles
        pitch_angles = []
        yaw_angles = []
        database = []
        for idx, (pitch_yaw) in enumerate(pitch_yaw_angles):
            pitch = int(pitch_yaw[0])
            yaw = int(pitch_yaw[1])

            # Set robot state and get link to world transformations
            robot.set_state(pitch, yaw, joint_pos)
            link_H_world = robot.compute_all_link_H_world()

            # Compute relative wind velocity vector
            wind_velocity = fn.compute_wind_velocity(pitch, yaw, WIND_INTENSITY)

            # Import fluent data from all surfaces
            flow.import_data(data_path, config, pitch, yaw)
            flow.transform_data(link_H_world, airspeed=17.0, air_dens=1.225)
            flow.reorder_data()
            flow.assign_global_data()

            # Graph Data Transformation
            edge_index = torch.tensor(flow.edges, dtype=torch.long).t().contiguous()
            wind_velocities = np.tile(wind_velocity, (flow.nodes.shape[0], 1))
            x = np.hstack((wind_velocities, flow.nodes, flow.face_normals, flow.areas))
            y = np.hstack((flow.press_coeff[:, None], flow.fric_coeff))
            x = torch.tensor(x, dtype=torch.float)
            y = torch.tensor(y, dtype=torch.float)
            data = Data(x=x, y=y, edge_index=edge_index)

            # Save data in database
            database.append(data)
            pitch_angles.append(pitch)
            yaw_angles.append(yaw)

            # Print progress
            print(
                f"{config} configuration progress: {idx+1}/{len(pitch_yaw_angles)}",
                end="\r",
                flush=True,
            )

        # Assign dataset variables
        dataset[config] = {
            "pitch_angles": np.array(pitch_angles).astype(np.float16),
            "yaw_angles": np.array(yaw_angles).astype(np.float16),
            "database": database,
        }

        # Save compressed dataset using pickle
        with open(str(dataset_dir / f"ironcub-{config}-dual-graph.npz"), "wb") as f:
            pickle.dump(dataset[config], f, protocol=4)
        print(f"Dual graph dataset for {config} configuration saved.")


if __name__ == "__main__":
    main()
