"""
Author: Antonello Paolino
Date: 2025-01-31
Description:    This code generates a dataset of the aerodynamic flow variables
                acting on the surface of the iRonCub robot as point cloud data.
                The dataset is structured to be used for training a Multi-Layer
                Perceptron (MLP) learning architecture.
"""

import numpy as np
from pathlib import Path
from resolve_robotics_uri_py import resolve_robotics_uri

import src.functions as fn
from src.robot import Robot
from src.flow import FlowImporter

WIND_INTENSITY = 17.0  # Fixed simulation wind intensity
DYN_PRESSURE = 0.5 * 1.225 * WIND_INTENSITY**2  # Dynamic pressure at S/L 17 m/s


def main():
    # Initialize robot object
    robot_name = "iRonCub-Mk3"
    urdf_path = str(resolve_robotics_uri("package://iRonCub-Mk3/model.urdf"))
    robot = Robot(robot_name, urdf_path)

    # Create a dataset output directory
    dataset_dir = Path(__file__).parents[0] / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Get the path to the mesh and data
    mesh_dir = input("Enter the path to the fluent msh directory: ")
    mesh_path = Path(str(mesh_dir).strip())
    data_dir = input("Enter the path to the fluent dtbs directory: ")
    data_path = Path(str(data_dir).strip())

    # Get hovering joint configuration
    joint_config_file = list(data_path.parent.rglob("joint-configurations.csv"))[0]
    joint_configs = np.genfromtxt(joint_config_file, delimiter=",", dtype=str)
    joint_pos = joint_configs[joint_configs[:, 0] == "hovering"][0, 1:].astype(float)

    # Import mesh data
    flow = FlowImporter()
    robot.set_state(0, 0, joint_pos * np.pi / 180.0)
    link_H_world_dict = robot.compute_all_link_H_world()
    flow.import_target_mesh(mesh_path, robot.surface_list, link_H_world_dict)

    # Visualize target mesh
    # fn.visualize_pointcloud(flow.w_nodes, flow.w_nodes[:, 0])

    # Get configuration names and joint configurations
    files = [file.name for file in data_path.rglob("*.dtbs") if file.is_file()]
    configs = sorted(list(set([file.split("-")[0] for file in files])))
    joint_config_file = list(data_path.parent.rglob("joint-configurations.csv"))[0]
    joint_configs = np.genfromtxt(joint_config_file, delimiter=",", dtype=str)

    # Start the cycle on configurations
    data = []
    for config in configs:
        joint_pos = joint_configs[joint_configs[:, 0] == config][0, 1:].astype(float)
        pitch_yaw_angles = fn.find_pitch_yaw_angles(config, files)
        config_data = {
            "data": [],
            "pitch_angles": [],
            "yaw_angles": [],
            "part_start_ids": flow.part_start_ids,
            "part_end_ids": flow.part_end_ids,
        }

        # Set robot state and get link_H_world
        robot.set_state(0, 0, joint_pos * np.pi / 180.0)
        link_H_world_dict = robot.compute_all_link_H_world()

        # Import source mesh
        flow.import_source_mesh(mesh_path, config, link_H_world_dict)

        # Visualize source mesh
        # fn.visualize_pointcloud(flow.w_nodes_src, flow.w_nodes_src[:, 0])

        # Start the cycle on simulations
        sim_num = len(pitch_yaw_angles)
        for idx, pitch_yaw_angle in enumerate(pitch_yaw_angles):
            pitch = int(pitch_yaw_angle[0])
            yaw = int(pitch_yaw_angle[1])

            # Set robot state and get link_H_world and base_H_world
            robot.set_state(pitch, yaw, joint_pos * np.pi / 180.0)
            link_H_world_dict = robot.compute_all_link_H_world()
            base_H_world = robot.compute_link_H_world(robot.base_link)

            # Import and transform simulation data
            flow.import_fluent_simulation_data(
                data_path, config, pitch, yaw, link_H_world_dict
            )
            world_H_link_dict = robot.compute_all_world_H_link()
            flow.interpolate_fluent_simulation_data(world_H_link_dict)
            flow.transform_data_to_base_link(base_H_world, DYN_PRESSURE)

            wind_velocity = fn.compute_wind_velocity(pitch, yaw, WIND_INTENSITY)
            wind_velocities = np.tile(wind_velocity, (flow.b_nodes.shape[0], 1))
            joint_pos_mat = np.tile(joint_pos, (flow.b_nodes.shape[0], 1))

            # Assemble single simulation data
            sim_data = np.hstack(
                (
                    wind_velocities,
                    joint_pos_mat,
                    flow.b_nodes,
                    flow.b_face_normals,
                    flow.press_coeff.reshape(-1, 1),
                    flow.b_fric_coeff,
                    flow.areas.reshape(-1, 1),
                )
            )

            # Append data to current configuration data
            config_data["data"].append(sim_data.astype(np.float32))
            config_data["pitch_angles"].append(pitch)
            config_data["yaw_angles"].append(yaw)

            # Visualize point cloud for debugging purposes
            # fn.visualize_pointcloud(flow.b_nodes, flow.press_coeff)

            # Print progress
            print(
                f"{config} configuration progress: {idx+1}/{sim_num}",
                end="\r",
                flush=True,
            )

        # Save compressed dataset using compressed numpy
        np.savez_compressed(dataset_dir / f"ironcub-{config}.npz", data=config_data)
        print(f"Dataset for {config} configuration saved.")

        # Append config_data to the global dataset
        data.extend(config_data)

    # Save compressed dataset using compressed numpy
    np.savez_compressed(dataset_dir / "ironcub-full.npz", data=data)
    print("Full dataset saved.")


if __name__ == "__main__":
    main()
